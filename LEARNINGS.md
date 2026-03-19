# LEARNINGS — NewsForge Build

Technical decisions, failures, and insights from building a 7-agent LangGraph research pipeline.

## Why LangGraph Over LangChain Chains

LangChain chains are linear: A → B → C → done. The Critic→Writer revision loop needs cycles — if the report scores below 0.70, route back to Writer with feedback notes, then re-evaluate. This is a conditional edge that creates a cycle in the execution graph.

LangGraph's `StateGraph` handles this natively with `add_conditional_edges`. The router function inspects state and returns "revise" or "done". Chains cannot express this.

When to use each: if your workflow is strictly linear (retrieval → generation → output), a chain is simpler and sufficient. If you need cycles, branching, or interrupt/resume, you need a graph.

## State Schema as Architecture Document

Designed the `NewsForgeState` TypedDict for all 7 agents before writing any agent code. This forced thinking about data contracts upfront: which agent produces `subtasks`, which agent consumes them, what fields does `analysis` contain.

Key discovery: `Annotated[list, operator.add]`. When a node returns `{"subtasks": [new_subtask]}`, LangGraph *appends* to the existing list instead of replacing it. Without this annotation, the second write to `subtasks` would overwrite the first. This is critical for fields that accumulate across multiple nodes (subtasks, search_results, scraped_content, errors).

Plain `Optional[str]` fields (like `draft_report`) get replaced on each write — correct for Writer revisions where the new draft should overwrite the old one.

Changed this approach after a previous RAG project where state was an afterthought. That project required three state schema rewrites as new agents were added. NewsForge: zero rewrites.

## ReAct Pattern — When It Adds Value

Planner uses ReAct because it needs to evaluate its own output quality. First pass generates subtasks + a self-assessed `coverage_score`. If coverage is below 0.70, the `coverage_gaps` field tells the LLM what angles are missing, and it refines. This catches blind spots like "you covered treatment but not prevention" on a healthcare topic.

Analysis uses ReAct because cross-source synthesis benefits from iteration. First pass extracts themes from the corpus. Second pass (if triggered) focuses on areas flagged in `coverage_notes`, often finding contradictions or nuances missed initially.

Writer does NOT use ReAct. Structured generation against a template is one-shot by nature — the analysis is already validated, and the Writer's job is formatting and prose. Adding a ReAct loop here would be over-engineering with no quality improvement.

In practice, coverage score hit 0.80 on iteration 1 for most topics. The ReAct loop rarely needed more than 1 pass, but the 2-3 times it did iterate, it caught genuine gaps.

## Checkpointing From Day One

Previous RAG project: Groq rate limits hit at topic 22/30 during a benchmark run. No checkpointing. Restarted from scratch. Lost 45 minutes of API calls and compute.

NewsForge: built SQLite checkpointer on day 1, before writing any agent code. Every node's output is persisted after completion. If the pipeline crashes at Analysis, it resumes from Analysis — not from Planner.

The resume key is simple: `thread_id = research_id`. LangGraph's `SqliteSaver` handles the rest. During benchmark development, the pipeline crashed 3 times (rate limits, network issues). Resumed each time with zero data loss.

The checkpointer also enables Human-in-the-Loop: `interrupt()` persists state to SQLite, and `Command(resume=...)` loads it back even after a server restart.

## Multi-Model Routing Evolution

This went through three iterations:

**Phase 1:** `llama-3.3-70b-versatile` for everything. Hit 100K TPD (tokens per day) limit after ~12 pipeline runs. Unusable for a 10-topic benchmark.

**Phase 2:** Moved to `qwen/qwen3-32b`. Better TPD budget but only 6K TPM (tokens per minute). Analysis agent sends ~6500 tokens per iteration — hit the per-minute limit on the first complex topic.

**Phase 3 (final):** Two-pool routing. Pool A: Scout 17B (30K TPM) for Planner, Analysis, Critic. Pool B: 8B instant (6K TPM) for Writer, Judge. Each pool has independent rate limits. 10-topic benchmark uses 20% of Pool A's daily budget and 16% of Pool B's.

Critic was originally in Pool B with Writer. But Writer (5500 tokens) + Critic (2500 tokens) exceeded Pool B's 6K TPM in rapid succession. Moved Critic to Pool A where 30K TPM absorbs it easily.

Key insight: TPM matters as much as TPD in agentic systems. A single large prompt can exhaust the per-minute budget even if the daily budget is fine. Two pools with independent limits prevent this.

## Benchmark Design

10 topics designed to stress different failure modes:

- **Easy** (healthcare AI, remote work, Gen Z mental health): Well-covered topics with abundant sources. Tests baseline quality.
- **Medium** (climate change, crypto regulation, social media): Nuanced topics with conflicting viewpoints. Tests Analysis agent's contradiction detection.
- **Hard** (quantum computing, CRISPR, fusion energy): Technical topics with limited accessible sources. Tests graceful degradation when scraper can't get full content.

Hard topics scored 7.9 vs easy 8.3 — a 5% drop, not a cliff. The pipeline degrades gracefully because the Writer can produce a coherent report even from partial analysis, and the Critic catches structural issues.

LLM-as-judge is separate from Critic intentionally. Critic scores on internal dimensions: factual accuracy, coherence, citation quality (0-1 scale, pass threshold 0.70). Judge scores on external dimensions: research depth, source diversity, topic coverage (0-10 scale, pass threshold 6.0). Critic consistently scored 0.85 on reports that Judge scored 7.5. Both metrics are needed — they measure different things.

## Scraper Reliability

httpx succeeds on ~74% of URLs. Failures come from:
- Paywalls (WSJ, FT, Bloomberg — return 403)
- JS-rendered pages (React SPAs — return empty HTML)
- Bot detection (Cloudflare challenges — return 403 or CAPTCHA HTML)
- Rate limiting (some sites block rapid sequential requests)

BeautifulSoup's fallback chain is critical: try `<article>` first, then `<main>`, then `div.content` / `div.article-body`, then `<body>`, then raw soup. Without this chain: 0% success on many news sites that use non-standard markup.

The 74% rate is acceptable for a portfolio project. Production fix: Jina Reader API (processes JS, bypasses most blocks, ~95% success) or Firecrawl.

## Analysis Binary Encoding Bug

Llama 4 Scout occasionally returns compressed binary data instead of JSON when the input prompt contains mixed-encoding content (common in scraped academic papers and PDFs). The output looks like: `}▯|▯}▯|▯}▯|...`

Root cause: mixed UTF-8/Latin-1/binary encoding in scraped academic content from sites like arxiv.org and pubmed. The model's tokenizer gets confused and produces garbage output.

Fix applied: 8K character corpus limit per Analysis iteration. This caps the input size and reduces the chance of hitting the encoding bug. Also added binary response detection — if the output contains `▯` patterns, the agent retries with a smaller corpus.

Production fix: pre-clean all scraped content with Jina Reader (normalizes encoding), or use llama-3.3-70b-versatile for Analysis (larger context window, more robust tokenizer).

## Non-Blocking API Design

Original implementation: `POST /research` called `pipeline.invoke()` synchronously. Pipeline takes 60-100 seconds. HTTP timeout = frontend hangs, Streamlit shows "running" forever.

Fix: background thread + polling. POST /research starts a `threading.Thread(target=_run_pipeline_background)` and returns `research_id` immediately. Frontend polls `GET /research/{id}/status` every 2 seconds. When status becomes `awaiting_approval`, the HITL UI appears.

State tracking uses an in-memory `active_runs` dict. This works for a portfolio project but loses state on server restart. Production fix: Redis or PostgreSQL.

The same pattern is used by every production AI API: OpenAI's batch endpoint returns a job ID, Anthropic's Messages API returns an `id` for streaming. Non-blocking + polling is the standard for long-running inference.

## What I Would Do Differently

1. **Design API as async from day 1** — retrofitting non-blocking was straightforward but the sync-first code had to be restructured. Starting async avoids this.
2. **Use Jina Reader instead of httpx for scraping** — would push success rate from 74% to ~95% and eliminate the binary encoding bug by normalizing content before Analysis.
3. **Run benchmark on cloud** — local runs hit free tier limits. A cloud runner with a dedicated Groq key would complete 10 topics without rate limit pauses.
4. **Add prompt versioning earlier** — Langfuse supports prompt metadata. I should have tagged each prompt version from the start instead of adding it midway through development.
5. **Separate benchmark API key from development key** — sharing a key means benchmark runs consume quota needed for development iteration.

## Production Recommendations

If deploying this system to production:

1. **Replace SQLite checkpointer with PostgreSQL** — LangGraph supports `PostgresSaver` natively. Better concurrency, no file locking issues.
2. **Replace httpx scraper with Jina Reader API** — ~95% success rate, handles JS rendering, normalizes encoding.
3. **Use llama-3.3-70b-versatile for Analysis** — best quality for cross-source synthesis, more robust tokenizer for mixed-encoding content.
4. **Add Redis for `active_runs` state** — currently in-memory dict, lost on server restart. Redis persists across restarts and supports TTL for automatic cleanup.
5. **Add authentication on /research endpoint** — currently open. API key header with env-var toggle for dev mode.
6. **Set up alerting on pipeline failures** — CloudWatch/Datadog alerts when error rate exceeds threshold or latency spikes.
7. **Implement WebSocket streaming** — replace polling with real-time pipeline events for lower latency UI updates.
