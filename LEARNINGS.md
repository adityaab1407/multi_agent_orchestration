# Learnings

Technical insights and lessons learned during the development of NewsForge.

---

## 1. LangGraph vs. LangChain Chains

**Problem:** LangChain's LCEL chains are linear — no loops, no branching.

**Solution:** LangGraph's `StateGraph` gives explicit control over nodes, edges, and conditional routing. This is essential for the Critic → Writer revision loop (Phase 2) where the graph must conditionally re-enter the Writer node.

**Key pattern:**
```python
graph.add_node("planner", planner_node)
graph.add_node("search", search_node)
graph.add_edge("planner", "search")
# Phase 2: conditional edge for revision loop
# graph.add_conditional_edges("critic", should_revise, {"revise": "writer", "pass": END})
```

---

## 2. SqliteSaver API — `from_conn_string` is a Context Manager

**Problem:** `SqliteSaver.from_conn_string("data/checkpoints.db")` returns an **async context manager** in newer `langgraph-checkpoint-sqlite` versions. Passing it directly to `graph.compile(checkpointer=...)` fails because the graph never enters the context.

**Solution:** Use a raw `sqlite3.Connection` and pass it directly:
```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
pipeline = graph.compile(checkpointer=checkpointer)
```

**Lesson:** Always check the actual return type when library APIs change between versions.

---

## 3. Langfuse v3 — The API Changed Completely

**Problem:** Langfuse v2 had `langfuse.trace()`, `.span()`, `.event()`, and `span.end(output=...)`. In Langfuse v3, **all of these are removed**.

**Solution:** Use the new context manager pattern:
```python
from langfuse import Langfuse, observe

langfuse_client = Langfuse()

# Create a trace using start_as_current_observation
with langfuse_client.start_as_current_observation(
    name="planner-trace",
    input={"topic": topic},
) as trace:
    # Nested span
    with langfuse_client.start_as_current_observation(
        name="llm-call",
    ) as span:
        result = llm.invoke(prompt)
        span.update(output={"result": result})
    
    trace.update(output={"subtasks": subtasks})
```

**Key differences:**
- `span.end(output=...)` → `span.update(output=...)`
- `.trace()` → `start_as_current_observation()`
- `trace.create_event()` still works for lightweight events

---

## 4. ReAct Pattern — Multi-Iteration Planning Matters

**Problem:** Single-shot prompting for subtask decomposition produces inconsistent, incomplete plans.

**Solution:** ReAct loop with coverage scoring:
1. LLM generates plan + `coverage_score` (0.0–1.0)
2. If score < threshold (0.7), LLM gets its own output back with instruction to improve
3. Max 3 iterations, converges quickly

**Result:** Plans consistently have 4–6 well-reasoned subtasks with no gaps. The self-critique mechanism catches obvious omissions the LLM would otherwise miss.

---

## 5. Pydantic V2 — LLM Safety Net

**Problem:** LLMs output JSON with wrong types, missing fields, extra fields — silently corrupting downstream state.

**Solution:** Parse every LLM response through Pydantic V2 models immediately:
```python
class SubtaskSchema(BaseModel):
    subtask_id: str
    title: str
    priority: int
    # ...

# Validate immediately after LLM response
data = json.loads(llm_response)
subtasks = [SubtaskSchema.model_validate(s) for s in data["subtasks"]]
```

**Benefit:** Errors surface instantly with clear messages instead of propagating silently.

---

## 6. Python 3.14 Compatibility Warnings

**Observation:** Running on Python 3.14 triggers Pydantic V1 deprecation warnings from `langchain-core` (which internally still references Pydantic V1 compatibility layer). These are **cosmetic only** — everything works correctly.

**Action:** Suppress with `warnings.filterwarnings` if needed, or wait for `langchain-core` to fully migrate.

---

## 7. Port Conflicts on Windows

**Problem:** Port 8000 was blocked (another process or Windows Defender).

**Solution:** Switched FastAPI to port 8080. Lesson: always make the port configurable:
```python
# backend/main.py uses --port flag
# frontend/app.py uses BACKEND_URL variable
# docker-compose.yml maps ports explicitly
```

---

## 8. Tavily Client — Retry with Exponential Backoff

**Problem:** Tavily API returns 429 (rate limit) when firing multiple queries in rapid succession.

**Solution:** Exponential backoff with 3 retries:
```python
for attempt in range(max_retries):
    try:
        return client.search(query=query, max_results=5)
    except Exception as e:
        if "429" in str(e):
            time.sleep(min(2 ** attempt, 60))
        else:
            raise
```

---

## 9. State Schema Design — `Annotated[list, operator.add]`

**Key insight:** LangGraph's state merge uses `operator.add` for list fields. This means:
- Each node appends to the list (not replaces)
- Returning `{"subtasks": new_subtasks}` from a node **extends** the existing list
- This is perfect for accumulating subtasks and search results across nodes

```python
class NewsForgeState(TypedDict):
    subtasks: Annotated[list[Subtask], operator.add]
    search_results: Annotated[list[SearchResult], operator.add]
```

---

## 10. Windows Venv Activation in PowerShell

**Gotcha:** On Windows, venv activation requires the `&` call operator:
```powershell
& ".\.multi_agent\Scripts\Activate.ps1"
```

Without `&`, PowerShell treats the path as a string, not a command.

---

## 11. Model Routing and Rate Dispersion

Choosing one model for all agents is the obvious first approach — and the wrong one for production agent systems. Here is the evolution:

**Phase 1: llama-3.3-70b everywhere**
Problem: 100K TPD exhausted after 1 pipeline run + benchmark. Fine for demos, not for evaluation.

**Phase 2: qwen/qwen3-32b for reasoning**
Problem: 6K TPM. Analysis agent sends ~6500 tokens per call — single call saturates per-minute budget.

**Phase 3: Two-pool routing**
- Pool A (Scout 17B): Planner, Analysis
- Pool B (8B Instant): Writer, Critic, Judge
  Problem discovered: Writer(5500t) + Critic(2500t) fire back-to-back = 8000 tokens in under a minute, exceeding 8B's 6K TPM limit.

**Phase 4: Rebalanced routing (current)**
- Pool A (Scout 17B, 30K TPM): Planner, Analysis, Critic
  Rationale: 30K TPM handles all three comfortably. Critic moved here to avoid TPM collision with Writer on Pool B.
- Pool B (8B Instant, 6K TPM): Writer, Judge
  Rationale: Writer alone (5500t) fits under 6K TPM. Judge runs separately during benchmark, never back-to-back with Writer.

**Key insight:** In multi-agent systems, per-minute (TPM) limits matter as much as daily (TPD) limits. A single large Analysis call can exhaust TPM even when daily quota is abundant. Model selection must account for the token profile of each individual agent call, not just the overall pipeline.

**Additional lesson:** Use dedicated API keys per project. Sharing keys across projects (NewsForge + RAG Benchmark) makes token tracking and rate limit attribution impossible. One key per project is non-negotiable.
