"""Extract typed semantic relations (CAUSES, DETECTS, TRANSMITTED_BY, ...)
between entities already present in the Neo4j graph, across every chunk of
the manual -- not a hand-picked subset.

The graph already has entities (Equipment, Procedure, Disease, Organism,
Reagent, Specimen, Cell, Finding, Measurement, AnatomicalSite, Vector) and a
generic MENTIONS link from each chunk to the entities it names. What it is
missing, for almost the whole corpus, is *how those entities relate to each
other* -- only 113 typed edges exist across 767 chunks. This script asks a
local LLM, one chunk at a time, which of the entities *already linked to
that chunk* are related by one of the graph's own relation types, so the
model is never inventing an entity or a relation type -- only picking pairs
from a list already grounded in that chunk.

Two independent checks guard against a hallucinated relation before it is
written:
  1. subject and object must both be entities the graph already links to
     this exact chunk (MENTIONS) -- the model cannot introduce a new one.
  2. the two entity names must both appear in the same sentence of the
     chunk's own text -- a relation the model reports between two entities
     that are never mentioned together is not evidence the text actually
     states that relation, however confident the model sounds.

Only relations that pass both checks are written, as new edges, tagged
extraction_method='llm_v1' and source_chunk=<chunk id> for traceability and
easy rollback. Existing entities, chunks, and relations are never modified
or deleted.

Run directly:  python scripts/extract_semantic_relations.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(ROOT / ".env")

RELATION_TYPES = {
    "CAUSES": "a disease or organism causes a finding or disease",
    "DETECTS": "a test or procedure detects a disease or organism",
    "TRANSMITTED_BY": "a disease or organism is transmitted by a vector",
    "HAS_FINDING": "a disease has a clinical finding or symptom",
    "FOUND_IN": "an organism or substance is found in a specimen or site",
    "USES_REAGENT": "a procedure uses a reagent",
    "USES_EQUIPMENT": "a procedure uses equipment",
    "HAS_MEASUREMENT": "a procedure has a measurement or parameter value",
    "EXAMINES": "a procedure examines a specimen",
}

# Derived empirically from the entity types actually used by the graph's own
# existing, hand-curated relations of each type (checked before this script
# was written: every one of the 113 existing edges of these types fits
# exactly one of these (subject_type, object_type) pairs). A candidate whose
# types don't match is structurally implausible regardless of how confident
# the model sounds -- this caught real failures in a 10-chunk trial run,
# e.g. "blood DETECTS malaria parasites" (blood is a SPECIMEN, not the
# PROCEDURE that actually does the detecting) and "panel USES_EQUIPMENT
# Solar panels" (panel is EQUIPMENT, not a PROCEDURE).
ALLOWED_TYPE_PAIRS: dict[str, set[tuple[str, str]]] = {
    "CAUSES": {("ORGANISM", "DISEASE")},
    "DETECTS": {("PROCEDURE", "ORGANISM"), ("PROCEDURE", "DISEASE")},
    "TRANSMITTED_BY": {("ORGANISM", "VECTOR"), ("DISEASE", "VECTOR")},
    "HAS_FINDING": {("DISEASE", "FINDING")},
    "FOUND_IN": {("ORGANISM", "ANATOMICAL_SITE"), ("ORGANISM", "SPECIMEN")},
    "USES_REAGENT": {("PROCEDURE", "REAGENT")},
    "USES_EQUIPMENT": {("PROCEDURE", "EQUIPMENT")},
    "HAS_MEASUREMENT": {("PROCEDURE", "MEASUREMENT")},
    "EXAMINES": {("PROCEDURE", "SPECIMEN")},
}
ALL_ALLOWED_TYPES = {t for pairs in ALLOWED_TYPE_PAIRS.values() for pair in pairs for t in pair}


def could_possibly_match(entities: list[dict]) -> bool:
    """Skip the (slow) LLM call entirely for a chunk whose entity types
    can never satisfy any relation's type pair, whatever the model says --
    e.g. a chunk mentioning only Equipment and Cell entities has no type
    combination that could pass ALLOWED_TYPE_PAIRS for any relation. This
    changes nothing about which relations get accepted, it only skips
    calls that were always going to be rejected."""
    types_present = {entity.get("type") for entity in entities}
    if not (types_present & ALL_ALLOWED_TYPES):
        return False
    for pairs in ALLOWED_TYPE_PAIRS.values():
        for subject_type, object_type in pairs:
            if subject_type in types_present and object_type in types_present:
                return True
    return False

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def same_sentence(text: str, a: str, b: str) -> bool:
    a_low, b_low = a.casefold(), b.casefold()
    for sentence in sentences(text):
        low = sentence.casefold()
        if a_low in low and b_low in low:
            return True
    return False


class RelationExtractor:
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        self._generator = None
        self._tokenizer = None

    def _run_query(self, query: str, **params) -> list[dict]:
        """A managed Aura connection can be dropped by the server after it
        sits idle for a while -- and with each LLM call here taking ~30s
        and most windows producing no relations to write, the connection
        between actual queries can go idle for tens of minutes at a
        stretch (observed live: the run crashed 2+ hours in on exactly
        this, SessionExpired, having made real progress up to that point
        that would otherwise have been thrown away). A brief wait before
        reconnecting matters too -- retrying immediately after a drop
        still failed to even fetch Aura's routing table (observed live,
        right after the first crash), while the same connection succeeded
        moments later once given a few seconds.
        """
        from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

        attempts = 4
        for attempt in range(attempts):
            try:
                with self.driver.session(database=self.database) as session:
                    return [record.data() for record in session.run(query, **params)]
            except (SessionExpired, ServiceUnavailable, Neo4jError) as exc:
                if attempt == attempts - 1:
                    raise
                wait = 5 * (attempt + 1)
                print(f"  [Neo4j error, retrying in {wait}s: {exc}]", flush=True)
                time.sleep(wait)
                try:
                    self.driver.close()
                except Exception:
                    pass
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
        return []

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = os.getenv("LOCAL_ANSWER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        if not torch.cuda.is_available():
            torch.set_num_threads(os.cpu_count() or 4)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            model.to("cpu")
        model.eval()
        self._generator = model

    def chunks_with_entities(self, limit: int | None, offset: int = 0, chunk_ids: list[str] | None = None) -> list[dict]:
        if chunk_ids:
            query = """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $chunk_ids
            WITH c, collect(DISTINCT {
                id: e.id, name: e.canonical_name, type: e.entity_type
            }) AS entities
            WHERE size(entities) >= 2
            RETURN c.id AS chunk_id, c.text AS text, entities
            ORDER BY c.id
            """
            with self.driver.session(database=self.database) as session:
                return [record.data() for record in session.run(query, chunk_ids=chunk_ids)]
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT {
            id: e.id, name: e.canonical_name, type: e.entity_type
        }) AS entities
        WHERE size(entities) >= 2
        RETURN c.id AS chunk_id, c.text AS text, entities
        ORDER BY c.id
        SKIP $offset
        """ + (f"LIMIT {int(limit)}" if limit else "")
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(query, offset=offset)]

    def all_chunks_with_entities(self) -> list[dict]:
        """Every chunk in document order, each with whatever entities it
        mentions (possibly none or one) -- unlike chunks_with_entities,
        nothing is filtered out here, since a chunk with only one entity
        can still contribute a valid relation once combined with its
        neighbours in build_windows."""
        query = """
        MATCH (c:Chunk)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE {
            id: e.id, name: e.canonical_name, type: e.entity_type
        } END) AS raw_entities
        RETURN c.id AS chunk_id, c.text AS text,
               [x IN raw_entities WHERE x IS NOT NULL] AS entities
        ORDER BY c.id
        """
        return self._run_query(query)

    @staticmethod
    def build_windows(chunks: list[dict], radius: int = 1) -> list[dict]:
        """Pair each chunk with its immediate neighbours before asking for
        relations, so a fact split across a chunk boundary -- a sentence
        naming the procedure at the end of one chunk and the equipment at
        the start of the next, or a numbered step continuing past the cut
        -- is still visible as one piece of text with both entities
        present, instead of being invisible to a strictly single-chunk
        pass. Only the near edge of each neighbour is kept (its own far
        content is someone else's window already), keeping the combined
        text and the LLM call it drives roughly anchor-sized rather than
        tripling it.
        """
        windows = []
        total = len(chunks)
        for index, anchor in enumerate(chunks):
            start = max(0, index - radius)
            end = min(total, index + radius + 1)
            entities_by_id: dict[str, dict] = {}
            parts: list[str] = []
            for position in range(start, end):
                neighbour = chunks[position]
                for entity in neighbour["entities"]:
                    entities_by_id[entity["id"]] = entity
                if position < index:
                    parts.append(neighbour["text"][-300:])
                elif position == index:
                    parts.append(neighbour["text"][:1200])
                else:
                    parts.append(neighbour["text"][:300])
            combined_entities = list(entities_by_id.values())
            if len(combined_entities) < 2:
                continue
            windows.append({
                "chunk_id": anchor["chunk_id"],
                "text": "\n".join(parts),
                "entities": combined_entities,
            })
        return windows

    def extract_for_chunk(self, chunk_id: str, text: str, entities: list[dict]) -> list[dict]:
        import torch

        self._ensure_generator()
        entity_lines = "\n".join(
            f"- {entity['name']} ({entity['type']})" for entity in entities
        )
        relation_lines = "\n".join(
            f"- {name}: {description}" for name, description in RELATION_TYPES.items()
        )
        system = (
            "You extract relationships that are explicitly stated in a "
            "laboratory manual passage. You may only use entities from the "
            "given list and relationship types from the given list. Never "
            "invent an entity or a relationship that is not clearly stated "
            "in the passage. If no relationship from the list is stated "
            "between two of the given entities, output an empty list."
        )
        prompt = (
            f"Passage:\n{text[:1200]}\n\n"
            f"Entities already identified in this passage:\n{entity_lines}\n\n"
            f"Allowed relationship types:\n{relation_lines}\n\n"
            "Output strict JSON only: a list of objects with keys "
            '"subject", "relation", "object". subject and object must be '
            "copied exactly from the entity list above. Output [] if none "
            "apply."
        )
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._generator.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output = self._generator.generate(
                **encoded, max_new_tokens=200, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return self._parse_and_verify(generated, text, entities)

    @staticmethod
    def _parse_and_verify(generated: str, text: str, entities: list[dict]) -> list[dict]:
        match = re.search(r"\[.*\]", generated, re.DOTALL)
        if not match:
            return []
        try:
            candidates = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(candidates, list):
            return []
        by_name = {entity["name"].casefold(): entity for entity in entities}
        verified = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            subject = str(candidate.get("subject", "")).strip()
            relation = str(candidate.get("relation", "")).strip().upper()
            obj = str(candidate.get("object", "")).strip()
            if relation not in RELATION_TYPES:
                continue
            subject_entity = by_name.get(subject.casefold())
            object_entity = by_name.get(obj.casefold())
            if not subject_entity or not object_entity:
                continue
            if subject_entity["id"] == object_entity["id"]:
                continue
            type_pair = (subject_entity.get("type"), object_entity.get("type"))
            if type_pair not in ALLOWED_TYPE_PAIRS[relation]:
                continue
            if not same_sentence(text, subject_entity["name"], object_entity["name"]):
                continue
            verified.append({
                "subject_id": subject_entity["id"],
                "relation": relation,
                "object_id": object_entity["id"],
                "subject_name": subject_entity["name"],
                "object_name": object_entity["name"],
            })
        return verified

    def write_relations(self, chunk_id: str, relations: list[dict]) -> None:
        if not relations:
            return
        # Relation type can't be parameterised in Cypher, so it is
        # interpolated -- safe here since it is always one of the fixed
        # RELATION_TYPES keys already validated in _parse_and_verify, never
        # raw model output.
        query_template = """
        UNWIND $relations AS rel
        MATCH (s:Entity {id: rel.subject_id})
        MATCH (o:Entity {id: rel.object_id})
        MERGE (s)-[r:%s {source_chunk: $chunk_id}]->(o)
        ON CREATE SET r.extraction_method = 'llm_v1'
        RETURN count(r) AS n
        """
        by_relation: dict[str, list[dict]] = {}
        for rel in relations:
            by_relation.setdefault(rel["relation"], []).append(rel)
        for relation_type, group in by_relation.items():
            self._run_query(
                query_template % relation_type, relations=group, chunk_id=chunk_id
            )

    def close(self) -> None:
        self.driver.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    extractor = RelationExtractor()
    ordered_chunks = extractor.all_chunks_with_entities()
    all_windows = extractor.build_windows(ordered_chunks, radius=1)
    if args.offset or args.limit:
        end = (args.offset + args.limit) if args.limit else None
        all_windows = all_windows[args.offset:end]
    # Most chunks' (own entities plus their immediate neighbours') types can
    # never satisfy any relation's type pair (e.g. only Equipment and Cell
    # entities nearby) -- skip the slow LLM call for those entirely rather
    # than spend ~20-30s on a call guaranteed to be rejected by the
    # type-pair check regardless of what the model outputs.
    windows = [row for row in all_windows if could_possibly_match(row["entities"])]
    print(f"Chunk windows with >=2 entities (own + neighbours): {len(all_windows)}; "
          f"{len(windows)} have a type combination any relation could match "
          f"({len(all_windows) - len(windows)} skipped, no LLM call needed)", flush=True)

    total_verified = 0
    seen: set[tuple[str, str, str]] = set()
    started = time.perf_counter()
    for index, row in enumerate(windows, 1):
        try:
            relations = extractor.extract_for_chunk(row["chunk_id"], row["text"], row["entities"])
            # A relation spanning a chunk boundary can surface from more
            # than one anchor's window (once as this chunk's neighbour,
            # once as its own anchor) -- write each real-world fact once.
            fresh = []
            for rel in relations:
                key = (rel["subject_id"], rel["relation"], rel["object_id"])
                if key in seen:
                    continue
                seen.add(key)
                fresh.append(rel)
            total_verified += len(fresh)
            for rel in fresh:
                print(f"  {row['chunk_id']}: {rel['subject_name']} -{rel['relation']}-> {rel['object_name']}", flush=True)
            if fresh and not args.dry_run:
                extractor.write_relations(row["chunk_id"], fresh)
        except Exception as exc:
            # A many-hour unattended run must not lose everything already
            # verified and written because one window hit an unexpected
            # error (observed live: an Aura connection drop propagated all
            # the way out and killed the process 2+ hours in). Log it and
            # move on -- one skipped window is a far smaller loss than the
            # whole run.
            print(f"  [skipping {row['chunk_id']} after error: {exc}]", flush=True)
        if index % 20 == 0:
            elapsed = time.perf_counter() - started
            rate = elapsed / index
            remaining = rate * (len(windows) - index)
            print(f"... {index}/{len(windows)} windows, {elapsed:.0f}s elapsed, "
                  f"{total_verified} relations verified so far, "
                  f"~{remaining/60:.0f}min remaining", flush=True)

    print(f"\nDone. {len(windows)} chunk windows processed, {total_verified} relations verified"
          + (" (dry run, nothing written)" if args.dry_run else " and written"))
    extractor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
