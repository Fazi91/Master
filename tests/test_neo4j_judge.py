"""Perturbation test for the Neo4j hallucination check.

For every benchmark question, the verified answer sentences that have a
Fact node in Neo4j are altered in controlled ways -- the kinds of error an
LLM rewrite makes -- and each altered sentence is judged against its Fact
exactly as the app judges an LLM sentence (webapp.pdf_direct_qa.
compare_with_fact with the app's own similarity checks). Every alteration
is produced by a general rule applied to whatever sentence it gets; no
question, sentence or answer is special-cased.

  unchanged         the source sentence itself          -> must be supported
  valid paraphrase  synonym substitutions               -> must be supported
  number changed    a quantity altered                  -> meaning changed
  condition dropped before/after, or, not ... removed   -> meaning changed
  qualifier changed a qualifier swapped for another     -> meaning changed
  qualifier dropped a qualifier removed                 -> information dropped
  statement dropped the last statement of the sentence  -> information dropped
  detail dropped    a descriptive word removed          -> information dropped
  claim added       an invented statement appended      -> unsupported claim
  steps swapped     a step judged against the next one  -> any problem

Results are reported separately for questions 1-19 (the rules were tuned
on manual feedback about these) and 20-39 (held out, never used for
tuning), so the held-out numbers show how the rules generalise.

Needs the Neo4j graph (with the Fact layer) and no LLM.
Run directly:  python tests/test_neo4j_judge.py
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from webapp.evaluation_app import EVALUATION_QUESTIONS, app, service
from webapp.pdf_direct_qa import (
    QUALIFIER_GROUPS,
    _premodifiers,
    compare_with_fact,
    fact_id,
    fact_qualifiers,
    words,
)

client = TestClient(app)

SYNONYMS = {
    "collect": "gather", "collected": "gathered", "place": "put", "specimen": "sample",
    "specimens": "samples", "approximately": "about", "ask": "instruct", "examine": "inspect",
    "examined": "inspected", "large": "big", "immediately": "at once", "quickly": "rapidly",
    "carefully": "cautiously", "thoroughly": "well", "check that": "ensure that",
}
CONDITION_EDITS = [
    (r"\bbefore\b", "after"), (r"\bafter\b", "before"), (r"\bnot\s+", ""), (r"\bnever\b", "always"),
    (r"\beither\s+", ""), (r",\s*or\b", ", and"), (r"\buntil\b", "after"), (r"\bat least\s+", ""),
    (r"\bat most\s+", ""), (r"\bunless\b", "if"), (r"\bif\b", "although"),
]
INVENTED = "Repeat the whole procedure with a fresh specimen from each member of the patient's family."
CATEGORIES = ("changed", "unsupported", "dropped")


def judge(fact: dict, text: str, list_items: str = "") -> dict[str, list[str]]:
    pdf = service().pdf
    changed, unsupported, dropped, review = compare_with_fact(
        fact, text, pdf.paraphrased, list_items, pdf.claim_covered, pdf.term_similarity,
    )
    return {"changed": changed, "unsupported": unsupported, "dropped": dropped, "review": review}


def replace_once(text: str, pattern: str, replacement: str) -> str | None:
    new = re.sub(pattern, replacement, text, count=1, flags=re.I)
    return new if new != text else None


def variants(body: str, fact: dict) -> list[tuple[str, str, str]]:
    """(name, altered sentence, expected category) for one source sentence."""
    out: list[tuple[str, str, str]] = [("unchanged", body, "supported")]

    paraphrase = body
    for word, synonym in SYNONYMS.items():
        paraphrase = re.sub(rf"\b{re.escape(word)}\b", synonym, paraphrase, flags=re.I)
    if paraphrase != body:
        out.append(("valid paraphrase", paraphrase, "supported"))

    for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", body):
        prefix = body[:match.start()].rstrip()
        if re.search(r"\b(?:fig(?:ure)?s?|tables?|sections?|no|page|reagent)\.?$", prefix, re.I):
            continue
        value = float(match.group(1))
        new_value = f"{value * 2:g}" if value else "3"
        out.append(("number changed", body[:match.start()] + new_value + body[match.end():], "changed"))
        break

    if fact.get("conditions"):
        for pattern, replacement in CONDITION_EDITS:
            altered = replace_once(body, pattern, replacement)
            if altered:
                out.append(("condition dropped", altered, "changed"))
                break

    qualifiers = sorted(fact_qualifiers(body) & set(fact.get("qualifiers") or []))
    if qualifiers:
        q = qualifiers[0]
        group = next((g for g in QUALIFIER_GROUPS if q in g), {q})
        other = "slowly" if "quickly" not in group else "carefully"
        swapped = replace_once(body, rf"\b{re.escape(q)}\b", other)
        if swapped:
            out.append(("qualifier changed", swapped, "changed"))
        removed = replace_once(body, rf"\s*\b{re.escape(q)}\b", "")
        if removed:
            out.append(("qualifier dropped", removed, "dropped"))

    claims = fact.get("claims") or []
    if len(claims) >= 2 and claims[-1] in body:
        out.append(("statement dropped", body.replace(claims[-1], "").strip(), "dropped"))

    for word in sorted(_premodifiers(body)):
        # only a word the sentence uses once is really gone once removed
        if len(re.findall(rf"\b{re.escape(word)}\b", body, flags=re.I)) != 1:
            continue
        removed = replace_once(body, rf"\b{re.escape(word)}\s+", "")
        if removed:
            out.append(("detail dropped", removed, "dropped"))
            break

    ending = "" if re.search(r"[.;:!?]$", body.rstrip()) else "."
    out.append(("claim added", f"{body.rstrip()}{ending} {INVENTED}", "unsupported"))
    return out


def main() -> int:
    started = time.perf_counter()
    graph = service().graph
    all_totals = {"tuned (Q1-19)": defaultdict(lambda: defaultdict(int)),
                  "held out (Q20-39)": defaultdict(lambda: defaultdict(int))}
    misses: dict[str, list[str]] = defaultdict(list)

    for item in EVALUATION_QUESTIONS:
        totals = all_totals["held out (Q20-39)" if item["id"] >= 20 else "tuned (Q1-19)"]
        response = client.post("/ask", json={"question": item["question"], "mode": "graph"})
        response.raise_for_status()
        evidence = [e for need in response.json().get("needs", []) for e in need.get("evidence", [])]
        ids = [fact_id(e["chunk_id"], e["text"].strip()) for e in evidence]
        facts = graph.facts_by_id(sorted(set(ids)))
        judged_steps: list[tuple[dict, str]] = []
        for index, (unit, identifier) in enumerate(zip(evidence, ids)):
            fact = facts.get(identifier)
            text = unit["text"].strip()
            body = re.sub(r"^\s*\d+[.)]\s+", "", text)
            if fact is None or len(words(body)) < 6:
                continue
            list_items = ""
            if text.endswith(":"):
                list_items = " ".join(
                    later["text"] for later in evidence[index + 1:]
                    if re.match(r"^\s*[-—•]", later["text"])
                )
            if re.match(r"^\s*\d+[.)]\s", text):
                judged_steps.append((fact, body))
            for name, altered, expected in variants(body, fact):
                verdict = judge(fact, altered, list_items)
                flagged = any(verdict[c] for c in CATEGORIES)
                ok = (not flagged) if expected == "supported" else bool(verdict[expected])
                totals[name]["n"] += 1
                totals[name]["ok"] += ok
                totals[name]["flagged"] += flagged
                totals[name]["review"] += bool(verdict["review"])
                if not ok and len(misses[name]) < 3:
                    misses[name].append(f"Q{item['id']}: {altered[:90]} -> {verdict}")
        for (fact_a, _body_a), (_fact_b, body_b) in zip(judged_steps, judged_steps[1:]):
            verdict = judge(fact_a, body_b)
            totals["steps swapped"]["n"] += 1
            ok = any(verdict[c] for c in CATEGORIES)
            totals["steps swapped"]["ok"] += ok
            totals["steps swapped"]["flagged"] += ok
            if not ok and len(misses["steps swapped"]) < 3:
                misses["steps swapped"].append(f"Q{item['id']}: {body_b[:90]}")
        print(f"Q{item['id']:>2} done", flush=True)

    order = ["unchanged", "valid paraphrase", "number changed", "condition dropped", "qualifier changed",
             "qualifier dropped", "statement dropped", "detail dropped", "claim added", "steps swapped"]
    for split, totals in all_totals.items():
        print(f"\n== {split}\n{'alteration':<18} {'n':>5} {'correct':>8} {'rate':>7}  note")
        for name in order:
            row = totals.get(name)
            if not row or not row["n"]:
                continue
            rate = 100 * row["ok"] / row["n"]
            note = (
                f"false alarms {row['flagged']}, review-only {row['review']}"
                if name in ("unchanged", "valid paraphrase") else f"any problem flagged {row['flagged']}"
            )
            print(f"{name:<18} {row['n']:>5} {row['ok']:>8} {rate:>6.1f}%  {note}")
    print("\nExamples of misses:")
    for name in order:
        for line in misses.get(name, []):
            print(f"  [{name}] {line}")
    print(f"\nTotal time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
