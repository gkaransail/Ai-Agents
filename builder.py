#!/usr/bin/env python3
"""
Builder Trust Scout — LangGraph agentic workflow
Evaluates Indian real estate builders via RERA status, project data, and review sentiment.

Usage:
    python agent.py "Lodha"
    python agent.py "Godrej Properties"

Env vars:
    ANTHROPIC_API_KEY  — required
    TAVILY_API_KEY     — optional; falls back to DuckDuckGo (free)
"""

import os, sys, re, json
from typing import TypedDict, Optional, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END


# ── Search tool bootstrap ─────────────────────────────────────────────────────

def _init_search():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        print("[init] Search: Tavily")
        return TavilySearchResults(max_results=5)
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        print("[init] Search: DuckDuckGo (free)")
        return DuckDuckGoSearchResults(max_results=5)
    except Exception as e:
        sys.exit(
            "[ERROR] No search tool found.\n"
            "Install: pip install duckduckgo-search\n"
            f"Detail: {e}"
        )

search = _init_search()

# ── LLM — Haiku 4.5 for budget efficiency ─────────────────────────────────────

llm = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512, temperature=0)


# ── State ─────────────────────────────────────────────────────────────────────

class BuilderState(TypedDict):
    builder_name:    str
    rera_id:         Optional[str]
    projects:        List[dict]    # [{name, status, delayed: bool}]
    reviews:         List[dict]    # [{source, sentiment, summary, complaint_keywords}]
    risk_score:      Optional[float]
    search_attempts: int


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ATTEMPTS = 3

# Queries escalate specificity on each retry
SEARCH_QUERIES = [
    "{n} RERA ID India registered projects 2024",
    "{n} developer RERA number MahaRERA TNRERA HRERA project list",
    "{n} real estate India projects delayed possession list complaints",
]

REVIEW_SOURCES = [
    ("{n} builder reviews ratings complaints housing.com",  "housing.com"),
    ("{n} developer reviews delayed possession magicbricks", "magicbricks.com"),
]


# ── Node: search ──────────────────────────────────────────────────────────────

def search_node(state: BuilderState) -> dict:
    """Search for RERA ID and project list. LLM extracts structured data."""
    n       = state["builder_name"]
    attempt = state.get("search_attempts", 0)
    query   = SEARCH_QUERIES[min(attempt, len(SEARCH_QUERIES) - 1)].format(n=n)

    print(f"\n[search #{attempt + 1}] {query}")

    try:
        raw = search.invoke(query)
        raw_str = json.dumps(raw) if not isinstance(raw, str) else raw
    except Exception as e:
        raw_str = f"search_error: {e}"

    prompt = (
        f'Extract from search results for builder "{n}".\n'
        "Return ONLY this JSON (no markdown, no explanation):\n"
        '{"rera_id": "P51900012345_or_null", "projects": [{"name": "...", '
        '"status": "completed|ongoing|delayed", "delayed": false}]}\n\n'
        f"Search results (first 2500 chars):\n{raw_str[:2500]}"
    )

    rera, projects = None, []
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        m = re.search(r'\{.*\}', resp.content, re.DOTALL)
        if m:
            data     = json.loads(m.group())
            rera     = data.get("rera_id") or None
            projects = data.get("projects") or []
            # Normalise: rera_id must be a non-empty string, not the literal "null"
            if isinstance(rera, str) and rera.lower() in ("null", "none", ""):
                rera = None
    except Exception as e:
        print(f"[search] LLM parse error: {e}")

    print(f"[search] RERA={'✓ ' + rera if rera else '✗ not found'} | "
          f"{len(projects)} project(s) found")

    return {
        "rera_id":         rera,
        "projects":        projects,
        "search_attempts": attempt + 1,
    }


# ── Node: review ──────────────────────────────────────────────────────────────

def review_node(state: BuilderState) -> dict:
    """Scrape and score review sentiment from 2 sources."""
    n       = state["builder_name"]
    reviews = []

    for q_tmpl, source in REVIEW_SOURCES:
        q = q_tmpl.format(n=n)
        print(f"\n[review] {source} ← '{q}'")

        try:
            raw = search.invoke(q)
            raw_str = json.dumps(raw) if not isinstance(raw, str) else raw
        except Exception as e:
            raw_str = f"search_error: {e}"

        prompt = (
            f'Analyse review sentiment for builder "{n}" from {source}.\n'
            "Return ONLY this JSON (no markdown):\n"
            f'{{"source": "{source}", "sentiment": "positive|negative|mixed|neutral", '
            '"summary": "one sentence", "complaint_keywords": ["delay", "quality"]}}\n\n'
            f"Data (first 1800 chars):\n{raw_str[:1800]}"
        )

        entry = {"source": source, "sentiment": "neutral",
                 "summary": "No data extracted", "complaint_keywords": []}
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            m = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if m:
                entry = json.loads(m.group())
        except Exception as e:
            print(f"[review] parse error: {e}")

        reviews.append(entry)
        print(f"[review] → {entry.get('sentiment','?').upper()} | {entry.get('summary','')}")

    return {"reviews": reviews}


# ── Node: analysis ────────────────────────────────────────────────────────────

def analysis_node(state: BuilderState) -> dict:
    """
    Risk Score formula:
        base       = delayed_projects / total_projects  (0.5 if unknown)
        sentiment  = (negative_reviews / total_reviews) × 0.20
        rera_miss  = 0.15 if no RERA ID found
        risk_score = min(base + sentiment + rera_miss, 1.0)
    """
    projects = state.get("projects", [])
    reviews  = state.get("reviews",  [])

    total   = len(projects)
    delayed = sum(1 for p in projects
                  if p.get("delayed") or p.get("status") == "delayed")

    base_ratio    = (delayed / total) if total else 0.5
    neg_count     = sum(1 for r in reviews if r.get("sentiment") == "negative")
    sentiment_adj = (neg_count / len(reviews)) * 0.20 if reviews else 0.10
    rera_penalty  = 0.0 if state.get("rera_id") else 0.15

    score = round(min(base_ratio + sentiment_adj + rera_penalty, 1.0), 3)

    print(f"\n[analysis] delayed={delayed}/{total} | "
          f"sentiment_adj={sentiment_adj:.2f} | "
          f"rera_penalty={rera_penalty:.2f} | "
          f"risk_score={score}")

    return {"risk_score": score}


# ── Edge: audit (conditional) ─────────────────────────────────────────────────

def audit_edge(state: BuilderState) -> str:
    """
    Gates progression from search → review.
    Loops back to search (with a different query) if:
      - RERA ID is missing  OR  project list is empty
      - AND we haven't hit MAX_ATTEMPTS yet
    """
    has_rera     = bool(state.get("rera_id"))
    has_projects = len(state.get("projects", [])) > 0
    attempts     = state.get("search_attempts", 0)

    if not (has_rera and has_projects) and attempts < MAX_ATTEMPTS:
        print(f"[audit] Insufficient data "
              f"(rera={has_rera}, projects={has_projects}). "
              f"Retry [{attempts}/{MAX_ATTEMPTS}]…")
        return "retry"

    print(f"[audit] Proceeding to review "
          f"(rera={has_rera}, projects={has_projects}, attempts={attempts})")
    return "proceed"


# ── Graph definition ──────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BuilderState)

    g.add_node("search",   search_node)
    g.add_node("review",   review_node)
    g.add_node("analysis", analysis_node)

    g.add_edge(START, "search")

    # Audit edge: search → search (retry) | search → review (proceed)
    g.add_conditional_edges(
        "search",
        audit_edge,
        {"retry": "search", "proceed": "review"},
    )

    g.add_edge("review",   "analysis")
    g.add_edge("analysis", END)

    return g.compile()


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(result: dict):
    name     = result.get("builder_name", "?")
    rera     = result.get("rera_id") or "Not Found"
    projects = result.get("projects", [])
    reviews  = result.get("reviews",  [])
    score    = result.get("risk_score") or 0.0

    total   = len(projects)
    delayed = sum(1 for p in projects
                  if p.get("delayed") or p.get("status") == "delayed")

    if score < 0.30:
        label = "LOW RISK     ✅"
    elif score < 0.60:
        label = "MEDIUM RISK  ⚠️ "
    else:
        label = "HIGH RISK    🚨"

    bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    sep   = "─" * 54

    print(f"\n{sep}")
    print(f"  BUILDER TRUST SCOUT — {name.upper()}")
    print(sep)
    print(f"  RERA ID      : {rera}")
    print(f"  Projects     : {total} found | {delayed} delayed")
    if projects:
        for p in projects[:6]:           # cap display at 6
            flag = "⚠ " if p.get("delayed") or p.get("status") == "delayed" else "✓ "
            print(f"    {flag}{p.get('name','?'):35} [{p.get('status','?')}]")
        if total > 6:
            print(f"    … and {total - 6} more")
    print(f"\n  Review Sentiment:")
    for r in reviews:
        kw = ", ".join(r.get("complaint_keywords", [])[:4])
        print(f"    ► {r.get('source','?'):22} {r.get('sentiment','?').upper():8}"
              f" | {r.get('summary','')}")
        if kw:
            print(f"      ↳ keywords: {kw}")
    print(f"\n  Risk Score   : {score:.3f}  [{bar}]")
    print(f"  Verdict      : {label}")
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(builder_name: str) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("[ERROR] ANTHROPIC_API_KEY not set.")

    print(f"\n{'═' * 54}")
    print(f"  Builder Trust Scout  →  {builder_name}")
    print(f"{'═' * 54}")

    app = build_graph()

    initial: BuilderState = {
        "builder_name":    builder_name,
        "rera_id":         None,
        "projects":        [],
        "reviews":         [],
        "risk_score":      None,
        "search_attempts": 0,
    }

    result = app.invoke(initial)
    print_report(result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python agent.py <builder_name>")
        print("Example: python agent.py Lodha")
        print('Example: python agent.py "Godrej Properties"')
        sys.exit(1)

    run(" ".join(sys.argv[1:]))
