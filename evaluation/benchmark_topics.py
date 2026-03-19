"""Curated benchmark topics for evaluating the NewsForge pipeline.

Each topic is designed to test different pipeline capabilities:
broad vs niche, technical vs general, easy scraping vs hard,
topics that produce contradictions vs consensus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BENCHMARK_TOPICS: list[dict] = [
    {
        "topic_id": "topic_001",
        "topic": "Impact of AI on healthcare in 2025",
        "category": "technology",
        "expected_difficulty": "medium",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers AI diagnostics, drug discovery, and patient care",
            "Mentions specific companies or tools (e.g. Google DeepMind, PathAI)",
            "Includes statistics on adoption rates or cost savings",
            "Addresses ethical concerns and bias in medical AI",
            "Discusses regulatory landscape (FDA approvals)",
        ],
        "known_challenges": [
            "Many sources are paywalled medical journals",
            "Hype-heavy coverage may lack concrete data",
        ],
    },
    {
        "topic_id": "topic_002",
        "topic": "Climate change adaptation strategies for coastal cities",
        "category": "science",
        "expected_difficulty": "medium",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers sea-level rise projections with specific numbers",
            "Mentions at least 3 specific cities and their strategies",
            "Includes infrastructure solutions (seawalls, green infrastructure)",
            "Discusses economic costs of adaptation vs inaction",
            "Addresses displacement and equity concerns",
        ],
        "known_challenges": [
            "Data varies widely by source and projection model",
            "Political framing may bias some sources",
        ],
    },
    {
        "topic_id": "topic_003",
        "topic": "State of quantum computing in 2025",
        "category": "technology",
        "expected_difficulty": "hard",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Explains current qubit counts and error rates",
            "Covers major players (IBM, Google, IonQ, PsiQuantum)",
            "Distinguishes hype from real milestones",
            "Addresses quantum advantage claims with evidence",
            "Discusses practical applications timeline",
        ],
        "known_challenges": [
            "Highly technical — scraper may miss nuance",
            "Press releases exaggerate breakthroughs",
            "Sources will contradict each other on timelines",
        ],
    },
    {
        "topic_id": "topic_004",
        "topic": "Economic impact of remote work on urban centers",
        "category": "business",
        "expected_difficulty": "easy",
        "expected_subtask_count": 4,
        "quality_criteria": [
            "Includes commercial real estate vacancy data",
            "Covers impact on local businesses (restaurants, retail)",
            "Discusses tax revenue changes for cities",
            "Compares different cities' experiences",
            "Mentions hybrid work trends and employer policies",
        ],
        "known_challenges": [
            "Data is widely available — quality filter matters",
            "Pre-2024 data may be outdated post-return-to-office push",
        ],
    },
    {
        "topic_id": "topic_005",
        "topic": "CRISPR gene editing latest breakthroughs 2025",
        "category": "health",
        "expected_difficulty": "hard",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers approved therapies (e.g. Casgevy for sickle cell)",
            "Discusses new CRISPR variants (base editing, prime editing)",
            "Includes clinical trial results with specific data",
            "Addresses safety concerns and off-target effects",
            "Mentions cost and accessibility of treatments",
        ],
        "known_challenges": [
            "Academic sources are often paywalled",
            "Requires distinguishing peer-reviewed results from press releases",
            "Technical terminology may confuse analysis agent",
        ],
    },
    {
        "topic_id": "topic_006",
        "topic": "Regulation of cryptocurrency markets globally",
        "category": "business",
        "expected_difficulty": "medium",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers US SEC actions and regulatory framework",
            "Includes EU MiCA regulation details",
            "Discusses China, Singapore, and UAE approaches",
            "Addresses stablecoin and DeFi regulation specifically",
            "Mentions impact on institutional adoption",
        ],
        "known_challenges": [
            "Regulatory landscape changes rapidly",
            "Crypto media is heavily biased",
            "Sources may contradict on regulatory intent",
        ],
    },
    {
        "topic_id": "topic_007",
        "topic": "Mental health crisis among Gen Z — causes and solutions",
        "category": "society",
        "expected_difficulty": "easy",
        "expected_subtask_count": 4,
        "quality_criteria": [
            "Cites specific statistics on anxiety and depression rates",
            "Covers social media as a contributing factor",
            "Discusses economic pressures (housing, student debt)",
            "Includes expert opinions from psychologists or researchers",
            "Mentions effective interventions and policy proposals",
        ],
        "known_challenges": [
            "Topic is widely covered — lots of sources available",
            "Risk of opinion-heavy content over data-driven analysis",
        ],
    },
    {
        "topic_id": "topic_008",
        "topic": "Supply chain resilience after COVID-19 lessons",
        "category": "business",
        "expected_difficulty": "easy",
        "expected_subtask_count": 4,
        "quality_criteria": [
            "Covers nearshoring and reshoring trends with examples",
            "Discusses technology solutions (AI forecasting, blockchain)",
            "Includes specific company case studies",
            "Addresses geopolitical risks (China dependency, chip shortage)",
            "Mentions inventory strategy changes (JIT vs JIC)",
        ],
        "known_challenges": [
            "Well-documented topic — should scrape easily",
            "Challenge is synthesis, not data collection",
        ],
    },
    {
        "topic_id": "topic_009",
        "topic": "Fusion energy progress and commercial viability",
        "category": "science",
        "expected_difficulty": "hard",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers NIF ignition milestone and follow-up results",
            "Discusses private fusion companies (Commonwealth, TAE, Helion)",
            "Includes specific timelines and investment figures",
            "Addresses engineering challenges beyond plasma physics",
            "Compares fusion timeline to renewable energy scaling",
        ],
        "known_challenges": [
            "Highly technical with lots of jargon",
            "Optimistic claims from startups vs skeptical scientists",
            "Limited number of authoritative sources",
        ],
    },
    {
        "topic_id": "topic_010",
        "topic": "Impact of social media algorithms on political polarization",
        "category": "society",
        "expected_difficulty": "medium",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Cites peer-reviewed studies on algorithmic amplification",
            "Covers multiple platforms (Meta, X/Twitter, TikTok, YouTube)",
            "Discusses filter bubbles vs echo chambers distinction",
            "Includes counterarguments (algorithms reflect, not create)",
            "Mentions regulatory proposals and platform responses",
        ],
        "known_challenges": [
            "Politically charged — sources will have strong viewpoints",
            "Academic vs journalistic sources will conflict",
            "Platform-specific data is hard to access",
        ],
    },
]


def get_topic_by_id(topic_id: str) -> dict | None:
    """Look up a single benchmark topic by its ID."""
    for topic in BENCHMARK_TOPICS:
        if topic["topic_id"] == topic_id:
            return topic
    return None


def get_topics_by_category(category: str) -> list[dict]:
    """Filter benchmark topics by category."""
    return [t for t in BENCHMARK_TOPICS if t["category"] == category]


def get_topics_by_difficulty(difficulty: str) -> list[dict]:
    """Filter benchmark topics by expected difficulty."""
    return [t for t in BENCHMARK_TOPICS if t["expected_difficulty"] == difficulty]
