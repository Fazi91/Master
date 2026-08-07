#start app http://127.0.0.1:8000/docs#/
#uvicorn webapp.main:app --reload
#start neo4j local aura, on ... robezan o bas connect kon
# passwort aura :nBLN1BRnoQHgnNnBEkYgCuOHkzYFqA3VGuvI3tWjl3g

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from scripts.retriever import hybrid_search_query, SEMANTIC_TO_NEO4J
from scripts.graph_client import GraphClient
from dotenv import load_dotenv
import os
import json
import re
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()
GRAPH = GraphClient()

# Strict thresholds for intent enforcement
DEFINITION_CONCEPT_THRESHOLD = 0.6  # Token-level concept coverage (synonym-aware)

# Lightweight synonym map (token-level) to avoid exact phrase dependence
SYNONYM_MAP = {
    "biosafety": ["bsl", "bio", "bio-safety"],
    "bsl": ["biosafety"],
    "level": ["lvl"],
    "laboratory": ["lab"],
    "lab": ["laboratory"],
    "cabinet": ["hood", "bsc"],
    "hood": ["cabinet", "bsc"],
    "autoclave": ["steam", "sterilizer", "pressure", "cooker"],
    "sterilization": ["decontamination", "disinfection"],
    "ppe": ["protective", "equipment"],
    "respirator": ["mask"],
}

# Load local LLM for final answer generation (using FLAN-T5 for better Q&A)
try:
    LLM_PIPE = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="cpu",
        max_length=150
    )
    LLM_READY = True
    print("[INFO] LLM loaded: google/flan-t5-small for Q&A")
except Exception as e:
    print(f"[WARNING] Could not load LLM model: {e}")
    LLM_READY = False

app = FastAPI(title="Simple RAG - Ask Questions")

# ==== BUILD STAMP (Verify new code is loaded) ====
import datetime
BUILD_TIMESTAMP = datetime.datetime.now().isoformat()
print(f"\n{'='*60}")
print(f"[BUILD STAMP] Started at: {BUILD_TIMESTAMP}")
print(f"[CONFIG] DEFINITION_CONCEPT_THRESHOLD = {DEFINITION_CONCEPT_THRESHOLD}")
print(f"[CONFIG] concept_match_mode = canonical+aliases+synonym_map")
print(f"[CONFIG] SYNONYM_MAP keys: {list(SYNONYM_MAP.keys())}")
print(f"{'='*60}\n")
# ===================================================

# Serve local image files under /static
IMAGES_DIR = Path("outputs/images/files").resolve()
if IMAGES_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(IMAGES_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    query: str


def _detect_question_intent(question: str) -> dict:
    """
    Classify question intent BEFORE retrieval AND extract core concepts as hard constraints.
    Returns: {"intent": str, "core_concepts": list, "entities": list, "concept_phrase": str}
    
    Intent types:
    - definition: "What is X?", "Define X"
    - explanation: "How does X work?", "Why X?"
    - comparison: "What is the difference between X and Y?"
    - procedure: "How to X?", "Steps to X"
    - fact_lookup: "Where is X?", "When X?"
    - yesno: "Is X?", "Can X?", "Does X?"
    """
    q_lower = question.lower().strip()
    
    # Extended stopwords for concept extraction
    STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", 
        "how", "why", "when", "where", "who", "which", "does", "do", "did", "can", "could",
        "should", "would", "will", "define", "definition", "meaning", "explain", "tell", 
        "me", "about", "please", "you", "i", "we", "they", "this", "that", "these", "those"
    }
    
    # Pattern-based classification (order matters - more specific first)
    patterns = {
        "definition": [
            r"what (?:is|are|does|do)\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"define\s+(.+?)(?:\?|$)",
            r"definition of\s+(.+?)(?:\?|$)",
            r"meaning of\s+(.+?)(?:\?|$)",
        ],
        "comparison": [
            r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$)",
            r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        ],
        "procedure": [
            r"how to\s+(.+?)(?:\?|$)",
            r"steps (?:to|for)\s+(.+?)(?:\?|$)",
            r"procedure (?:for|to)\s+(.+?)(?:\?|$)",
            r"process (?:of|for)\s+(.+?)(?:\?|$)",
        ],
        "explanation": [
            r"how (?:does|do)\s+(.+?)\s+work",
            r"why (?:is|does|do|are)\s+(.+?)(?:\?|$)",
            r"explain\s+(.+?)(?:\?|$)",
            r"reason (?:for|why)\s+(.+?)(?:\?|$)",
        ],
        "fact_lookup": [
            r"where (?:is|are|does|do)\s+(.+?)(?:\?|$)",
            r"when (?:is|are|does|do)\s+(.+?)(?:\?|$)",
            r"which\s+(.+?)(?:\?|$)",
            r"who\s+(.+?)(?:\?|$)",
        ],
        "yesno": [
            r"^(?:is|are|can|does|do|should|will|would)\s+(.+?)(?:\?|$)",
        ],
    }
    
    detected_intent = "fact_lookup"  # default
    concept_phrase = ""
    raw_concepts = []
    pattern_matched = False

    for intent, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, q_lower)
            if match:
                detected_intent = intent
                raw_concepts = [
                    group.strip()
                    for group in match.groups()
                    if group and group.strip()
                ]
                concept_phrase = " ".join(raw_concepts)
                pattern_matched = True
                break

        if pattern_matched:
            break

    # Recognize operational headings that describe laboratory procedures
    # without using explicit question forms such as "how to".
    if not pattern_matched:
        procedure_heading_patterns = [
            r"^(?:technique|method|procedure|process)\s+(?:for|of|to)\s+.+$",
            r"^(?:preparation|collection|examination|detection|identification|measurement|testing|staining|fixation|washing|incubation|sterilization|disinfection)\s+(?:of|for|with|using)\s+.+$",
            r"^(?:preparing|collecting|examining|detecting|identifying|measuring|testing|staining|fixing|washing|incubating|sterilizing|disinfecting)\s+.+$",
        ]

        if any(
            re.search(pattern, q_lower)
            for pattern in procedure_heading_patterns
        ):
            detected_intent = "procedure"
            concept_phrase = q_lower

    # Extract core concepts (nouns/entities) as HARD CONSTRAINTS
    def _extract_core_concepts(phrase: str) -> list:
        """Extract meaningful multi-word entities and single-word concepts"""
        # First: extract multi-word entities (2-3 consecutive important words)
        tokens = [t for t in re.findall(r'\b\w+\b', phrase.lower()) if t not in STOPWORDS]

        multi_word_entities = []
        for i in range(len(tokens) - 1):
            # Capture 2-word and 3-word phrases
            two_word = f"{tokens[i]} {tokens[i+1]}"
            multi_word_entities.append(two_word)
            if i < len(tokens) - 2:
                three_word = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                multi_word_entities.append(three_word)

        # Single meaningful tokens (length > 3 or known important terms)
        important_terms = {"bsl", "ppe", "who", "cdc", "fda"}
        single_concepts = [t for t in tokens if len(t) > 3 or t in important_terms]

        return multi_word_entities + single_concepts

    if not concept_phrase:
        # Extract from full question if no pattern match
        concept_phrase = q_lower

    core_concepts = _extract_core_concepts(concept_phrase)

    # Extract entities (domain-specific terms)
    DOMAIN_ENTITIES = [
        "biosafety", "bsl", "containment", "level", "risk group",
        "ppe", "protective equipment", "glove", "mask", "respirator", "gown",
        "autoclave", "sterilization", "disinfection",
        "laboratory", "reagent", "specimen", "culture",
        "refrigerator", "freezer", "storage",
        "microscope", "centrifuge", "incubator",
        "pregnancy test", "test kit", "diagnostic"
    ]

    entities = [e for e in DOMAIN_ENTITIES if e in q_lower]

    print(f"[INTENT] Question: '{question}' → Intent: {detected_intent}")
    print(f"[CONCEPTS] Core concepts (HARD CONSTRAINTS): {core_concepts[:5]}")
    print(f"[ENTITIES] Domain entities: {entities}")

    return {
        "intent": detected_intent,
        "core_concepts": core_concepts[:10],  # Top 10 concepts as hard constraints
        "entities": entities,
        "concept_phrase": concept_phrase,
    }


def _pick_first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def _paths_for_search():
    index_path = _pick_first([
        "outputs/text/minilm_index.faiss",
        "outputs/text/bge_index.faiss",
    ])
    meta_path = _pick_first([
        "outputs/text/minilm_meta.jsonl",
        "outputs/text/bge_meta.jsonl",
    ])
    chunks_path = _pick_first([
        "outputs/who_chunks_semantic.jsonl",
        "outputs/who_chunks_baseline.jsonl",
        "outputs/who_preview.json",
    ])
    return index_path, meta_path, chunks_path


def _faiss_only_search(query: str, topk: int, index_path: Path, meta_path: Path, allowed_ids=None):
    """
    FAISS search with optional ID restriction (graph-validated IDs only).
    
    Args:
        query: Search query
        topk: Number of results
        index_path: FAISS index file
        meta_path: Metadata file
        allowed_ids: List of allowed semantic IDs (graph-validated). If None, search all.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], normalize_embeddings=True)
    index = faiss.read_index(str(index_path))
    
    # Fetch more results if we need to filter
    search_k = topk * 5 if allowed_ids else topk
    D, I = index.search(q_emb.astype("float32"), search_k)
    
    id_map = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                id_map.append(obj.get("id"))
            except Exception:
                id_map.append(None)
    
    results = []
    for rank, idx in enumerate(I[0]):
        sem_id = id_map[idx] if idx < len(id_map) else None
        score = float(D[0][rank])
        if sem_id:
            # Filter by allowed IDs if provided
            if allowed_ids is None or sem_id in allowed_ids:
                results.append((sem_id, score))
                if len(results) >= topk:
                    break
    
    return results


def _map_semantic_to_text_ids(semantic_ids_scores):
    text_ids = []
    for sem_id, _ in semantic_ids_scores:
        tid = SEMANTIC_TO_NEO4J.get(sem_id)
        if tid:
            text_ids.append(tid)
    return text_ids


def _fetch_texts_by_ids(ids):
    if not ids:
        return []
    rows = GRAPH.fetch_text_chunks(ids)
    
    # CRITICAL: Neo4j returns results in ARBITRARY order
    # We must preserve the input order (sorted by relevance score)
    # Build a lookup dict and then reconstruct in original order
    lookup = {r["id"]: r for r in rows}
    
    ordered_results = []
    for tid in ids:
        if tid in lookup:
            r = lookup[tid]
            text_content = r.get("text", "")
            
            # SANITY CHECK: Validate known mappings
            if tid == "T_00059" and "refrigerator" not in text_content.lower():
                print(f"[ERROR] ID mapping corruption detected! T_00059 should contain 'refrigerator' but got: {text_content[:200]}")
            
            ordered_results.append({
                "id": r["id"], 
                "page": r.get("page"), 
                "content": text_content
            })
        else:
            print(f"[WARNING] Requested ID {tid} not found in Neo4j results")
    
    return ordered_results


def _verify_evidence_type_match(chunk_content: str, question_intent: str) -> tuple[bool, str]:
    """
    Verify that chunk contains evidence type matching the question intent.
    Returns: (matches: bool, reason: str)
    
    Evidence requirements by intent:
    - definition: must contain definitional language ("is defined as", "refers to", "means")
    - procedure: must contain steps/actions ("step", "should", "must", "procedure")
    - comparison: must mention comparative language ("difference", "compared to", "versus")
    - fact_lookup: must contain explicit values (numbers, dates, locations)
    - explanation: must contain explanatory language ("because", "due to", "reason")
    - yesno: must contain explicit affirmation/negation or conditions
    """
    content_lower = chunk_content.lower()
    
    EVIDENCE_PATTERNS = {
        "definition": {
            # Strict: definitional phrases OR explicit BSL-style phrasing
            "required": [
                " is defined as", " refers to ", " is a ", " are ", " means ", 
                " definition ", " is the ", " are the ", " known as "
            ],
            "reason": "definitional language",
            "check_func": lambda c: bool(re.search(r"\blevel\s+\d+\s+laborator", c))
        },
        "procedure": {
            "required": [
                "step ", "procedure", "method", "how to", "should be", 
                "must be", "process", "first", "then", "next", "finally"
            ],
            "reason": "procedural steps or instructions"
        },
        "comparison": {
            "required": [
                "difference", "differ", "compared", "versus", "vs", 
                "contrast", "unlike", "while", "whereas", "both"
            ],
            "reason": "comparative language"
        },
        "fact_lookup": {
            "required": [],  # Uses pattern matching instead
            "reason": "explicit facts or values",
            "check_func": lambda c: bool(re.search(r'\d+|\b(where|when|located|at)\b', c))
        },
        "explanation": {
            "required": [
                "because", "due to", "reason", "cause", "result", 
                "since", "therefore", "thus", "consequently"
            ],
            "reason": "explanatory language"
        },
        "yesno": {
            "required": [
                "yes", "no", "can", "cannot", "does", "does not", 
                "is ", "is not", "required", "optional", "mandatory"
            ],
            "reason": "explicit affirmation or negation"
        }
    }
    
    if question_intent not in EVIDENCE_PATTERNS:
        return True, "intent not restricted"  # Allow by default for unknown intents
    
    pattern_config = EVIDENCE_PATTERNS[question_intent]
    
    # Check custom function if provided
    if "check_func" in pattern_config:
        if pattern_config["check_func"](content_lower):
            return True, pattern_config["reason"]
    
    # Check required patterns
    if pattern_config["required"]:
        if any(pattern in content_lower for pattern in pattern_config["required"]):
            return True, pattern_config["reason"]
    
    # No match found
    return False, f"missing {pattern_config['reason']}"


def _score_definition_indicators(text: str) -> dict:
    """
    Score chunk for definition-related indicators.
    Returns: {"indicator_count": int, "indicators_found": [str], "definition_score": float}
    """
    indicators = {
        "containment": r"\bcontainment\b",
        "containment_level": r"\bcontainment level\b",
        "ppe": r"\b(ppe|personal protective equipment|protective gear|protective equipment)\b",
        "respiratory": r"\b(respiratory protection|respirator|breathing apparatus)\b",
        "restricted": r"\b(restricted access|restricted area|limited access|access control)\b",
        "biosafety_cabinet": r"\b(biosafety cabinet|safety cabinet|biological safety cabinet|bsc)\b",
        "decontamination": r"\b(decontamination|disinfection|sterilization|autoclave)\b",
        "aerosol": r"\b(aerosol|airborne|droplet|inhalation|aerosol generation)\b",
        "level_1": r"\b(level 1|bsl-?1|biosafety level 1)\b",
        "level_2": r"\b(level 2|bsl-?2|biosafety level 2)\b",
        "level_3": r"\b(level 3|bsl-?3|biosafety level 3)\b",
        "level_4": r"\b(level 4|bsl-?4|biosafety level 4)\b",
    }
    
    text_lower = text.lower()
    found = []
    for indicator_name, pattern in indicators.items():
        if __import__("re").search(pattern, text_lower, __import__("re").IGNORECASE):
            found.append(indicator_name)
    
    # Scoring: normalized by max 12 indicators
    definition_score = len(found) / 12.0
    
    return {
        "indicator_count": len(found),
        "indicators_found": found,
        "definition_score": definition_score
    }


def _bundle_evidence_chunks(chunks: list, core_concepts: list, max_bundle_size: int = 4) -> list:
    """
    Bundle chunks for better evidence presentation.
    Groups chunks by: same canonical concept, adjacent pages, same content_role.
    Returns: [{"chunks": [...], "concept": str, "pages": [...], "bundle_score": float, "indicators": [...]}]
    """
    import re
    
    if not chunks:
        return []
    
    # Enrich chunks with definition scores
    for chunk in chunks:
        if "definition_score" not in chunk:
            chunk.update(_score_definition_indicators(chunk.get("content", "")))
        if "role" not in chunk:
            # Fallback role assignment if not already present
            content = chunk.get("content", "")
            content_lower = content.lower()
            if any(p in content_lower for p in ["is defined as", "is classified as", "refers to", "means", "consists of"]):
                chunk["role"] = "definition"
            elif any(p in content_lower for p in ["procedure", "step", "follow these", "must be"]):
                chunk["role"] = "procedure"
            elif content.count("\n") > 5 or "|" in content:
                chunk["role"] = "table"
            elif any(p in content_lower for p in ["example", "for example", "such as"]):
                chunk["role"] = "example"
            else:
                chunk["role"] = "general"
    
    # Group chunks by concept and page proximity
    bundles_dict = {}
    for chunk in chunks:
        # Try to find a matching bundle based on concept and page
        chunk_page = chunk.get("page", 0)
        chunk_concept = chunk.get("core_concept", "general")  # Will use if available
        
        # Find or create bundle
        bundle_key = None
        for key, bundle in bundles_dict.items():
            # Check if chunk fits this bundle (same role, nearby pages, or same concept)
            if len(bundle["chunks"]) < max_bundle_size:
                bundle_pages = [c.get("page", 0) for c in bundle["chunks"]]
                avg_page = sum(bundle_pages) / len(bundle_pages) if bundle_pages else chunk_page
                page_dist = abs(chunk_page - avg_page)
                
                same_role = chunk.get("role") == bundle["chunks"][0].get("role")
                nearby_pages = page_dist <= 3  # Within 3 pages
                
                if nearby_pages or same_role:
                    bundle_key = key
                    break
        
        if bundle_key is None:
            bundle_key = f"bundle_{len(bundles_dict)}"
        
        if bundle_key not in bundles_dict:
            bundles_dict[bundle_key] = {
                "chunks": [],
                "concept": chunk_concept,
                "pages": set(),
                "roles": set(),
                "indicators_set": set(),
            }
        
        bundles_dict[bundle_key]["chunks"].append(chunk)
        bundles_dict[bundle_key]["pages"].add(chunk_page)
        bundles_dict[bundle_key]["roles"].add(chunk.get("role", "general"))
        
        # Collect all indicators found across bundle
        for ind in chunk.get("indicators_found", []):
            bundles_dict[bundle_key]["indicators_set"].add(ind)
    
    # Convert to list and score bundles
    bundles = []
    for bundle_key, bundle_data in bundles_dict.items():
        chunk_list = bundle_data["chunks"]
        
        # Score bundle by:
        # 1. Definition coverage (avg definition_score) - 40%
        # 2. Concept consistency (all chunks match canonical) - 30%
        # 3. Section cohesion (same role or nearby pages) - 30%
        
        avg_def_score = sum(c.get("definition_score", 0) for c in chunk_list) / max(1, len(chunk_list))
        
        # Concept consistency: check if chunks share core concepts or aliases
        has_consistent_concept = len(bundle_data["roles"]) <= 2  # Good if roles are consistent
        concept_consistency = 1.0 if has_consistent_concept else 0.5
        
        # Section cohesion: prefer single-page or nearby pages
        pages_list = list(bundle_data["pages"])
        page_spread = max(pages_list) - min(pages_list) if pages_list else 0
        section_cohesion = max(0, 1.0 - (page_spread / 10.0))  # Lower spread = higher cohesion
        
        bundle_score = (avg_def_score * 0.4) + (concept_consistency * 0.3) + (section_cohesion * 0.3)
        
        bundles.append({
            "chunks": chunk_list,
            "concept": bundle_data["concept"],
            "pages": sorted(list(bundle_data["pages"])),
            "roles": list(bundle_data["roles"]),
            "indicators": list(bundle_data["indicators_set"]),
            "bundle_score": bundle_score,
            "bundle_key": bundle_key,
        })
    
    # Sort by bundle score descending
    bundles.sort(key=lambda b: b["bundle_score"], reverse=True)
    
    return bundles


def _flatten_bundles(bundles: list, max_total_chunks: int = 10) -> list:
    """
    Flatten top-scoring bundles back into flat chunk list, maintaining order by bundle score.
    
    Args:
        bundles: List of bundle dicts from _bundle_evidence_chunks
        max_total_chunks: Maximum total chunks to return
    
    Returns:
        List of chunks sorted by bundle score, then within-bundle order
    """
    flat_chunks = []
    for bundle in bundles:
        chunks_to_add = bundle["chunks"][:max(1, max_total_chunks - len(flat_chunks))]
        flat_chunks.extend(chunks_to_add)
        if len(flat_chunks) >= max_total_chunks:
            break
    
    return flat_chunks[:max_total_chunks]


def _filter_by_evidence_type(chunks: list, question_intent: str) -> tuple[list, list]:
    """
    Filter chunks by evidence type matching.
    Returns: (matching_chunks, rejected_chunks_with_reasons)
    """
    if not chunks:
        return [], []
    
    matching = []
    rejected = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        matches, reason = _verify_evidence_type_match(content, question_intent)
        
        if matches:
            matching.append(chunk)
            print(f"[EVIDENCE] [OK] {chunk.get('id')} matches {question_intent} - has {reason}")
        else:
            rejected.append((chunk, reason))
            print(f"[EVIDENCE] [X] {chunk.get('id')} rejected - {reason}")
    
    return matching, rejected


# ============================================================================
# STAGE 4: 2-PASS SOFT RE-DISCOVERY LOOP - Query Expansion & Section Narrowing
# ============================================================================

def _expand_query_with_aliases(query: str, core_concepts: list, entities: list) -> dict:
    """
    PASS 2: Generate expanded query variants using canonical concept aliases and paraphrases.
    
    Returns: {
        "original": str,
        "expanded_queries": list[str],
        "alias_map": dict,  # Maps each canonical to its aliases
        "reasoning": str
    }
    """
    # Get synonym/alias expansions from graph client's concept normalization
    alias_map = {}
    expanded_queries = [query]  # Keep original
    
    # BSL-specific aliases
    bsl_aliases = {
        "biosafety": ["bsl", "bio-safety", "biological safety"],
        "biosafety level": ["bsl level", "bsl", "containment level"],
        "biosafety level 1": ["bsl-1", "bsl 1", "level 1"],
        "biosafety level 2": ["bsl-2", "bsl 2", "level 2"],
        "biosafety level 3": ["bsl-3", "bsl 3", "level 3"],
        "biosafety level 4": ["bsl-4", "bsl 4", "level 4"],
        "biosafety cabinet": ["bsc", "biological safety cabinet", "safety cabinet"],
        "ppe": ["protective equipment", "protection"],
        "autoclave": ["steam sterilization", "pressure cooker"],
        "sterilization": ["decontamination", "disinfection"],
    }
    
    for concept in core_concepts:
        concept_lower = concept.lower()
        # Check if concept has known aliases
        for key, aliases in bsl_aliases.items():
            if key in concept_lower or any(alias in concept_lower for alias in aliases):
                alias_map[key] = aliases
                for alias in aliases:
                    new_query = query.replace(concept, alias, 1)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
    
    # Paraphrase patterns for common intents
    intent_paraphrases = {
        "definition": ["What is meant by", "Describe", "Explain the concept of"],
        "procedure": ["How do you", "Steps for", "Process to"],
        "comparison": ["Differences in", "How does compare to"],
    }
    
    reasoning = f"Generated {len(expanded_queries)} query variants from {len(core_concepts)} core concepts"
    
    return {
        "original": query,
        "expanded_queries": list(set(expanded_queries))[:10],  # Cap at 10
        "alias_map": alias_map,
        "reasoning": reasoning
    }


def _run_pass2_discovery(
    query: str,
    core_concepts: list,
    entities: list,
    graph_validation: dict,
    index_path,
    meta_path,
    chunks_path,
    restricted_pages: list = None,
) -> dict:
    """
    PASS 2: Soft re-discovery with expanded queries and section narrowing.
    
    Strategy:
    1. Expand query with aliases/paraphrases
    2. If graph validated, restrict to graph-validated sections/pages
    3. Allow higher TopK for relaxed retrieval
    4. Re-validate with same strict rules
    
    Returns: {
        "success": bool,
        "expanded_queries": list[str],
        "candidate_chunks": list,
        "candidate_count": int,
        "restricted_to_pages": list,
        "pass2_reason": str,
        "log": list[str]
    }
    """
    log = []
    
    # Quick exit: if no graph validation at all, skip Pass 2
    if not graph_validation.get("exists"):
        log.append("[PASS2] Skipping: No graph validation available")
        return {
            "success": False,
            "expanded_queries": [],
            "candidate_chunks": [],
            "candidate_count": 0,
            "restricted_pages": [],
            "pass2_reason": "graph_unavailable",
            "log": log
        }
    
    # Step 1: Expand query
    expansion_result = _expand_query_with_aliases(query, core_concepts, entities)
    expanded_queries = expansion_result["expanded_queries"]
    log.append(f"[PASS2] Original: '{query}'")
    log.append(f"[PASS2] Expanded queries ({len(expanded_queries)}): {expanded_queries[:3]}")
    
    # Step 2: Identify restricted pages from graph validation
    restricted_pages = graph_validation.get("pages", []) if graph_validation.get("exists") else []
    log.append(f"[PASS2] Restricting search to {len(restricted_pages)} validated pages: {restricted_pages[:5]}")
    
    if not restricted_pages:
        log.append("[PASS2] No restricted pages available")
        return {
            "success": False,
            "expanded_queries": expanded_queries,
            "candidate_chunks": [],
            "candidate_count": 0,
            "restricted_pages": [],
            "pass2_reason": "no_section_candidates",
            "log": log
        }
    
    # Step 3: Run expanded queries with higher TopK (relaxed retrieval only in discovery)
    PASS2_TOPK = 20  # Moderate TopK to avoid excessive fetching
    candidate_chunks = []
    candidate_ids_seen = set()
    
    # Limit to first 5 expanded queries to avoid timeout
    for exp_query in expanded_queries[:5]:
        try:
            # Search with expanded query
            faiss_hits = _faiss_only_search(exp_query, PASS2_TOPK, Path(index_path), Path(meta_path), allowed_ids=None)
            
            # Map and fetch (limited to top 10)
            text_ids = _map_semantic_to_text_ids(faiss_hits[:10])
            chunks = _fetch_texts_by_ids(text_ids)
            
            # Filter to restricted pages if available
            if restricted_pages:
                chunks = [c for c in chunks if c.get("page") in restricted_pages]
            
            for chunk in chunks:
                chunk_id = chunk.get("id")
                if chunk_id not in candidate_ids_seen:
                    candidate_chunks.append(chunk)
                    candidate_ids_seen.add(chunk_id)
            
            log.append(f"[PASS2] Query '{exp_query[:30]}...': Found {len(chunks)} candidates in restricted pages")
            
            # Stop early if we have enough candidates
            if len(candidate_chunks) >= 5:
                log.append(f"[PASS2] Stopping early with {len(candidate_chunks)} candidates")
                break
        except Exception as e:
            log.append(f"[PASS2] Query '{exp_query[:30]}...': Error - {str(e)[:40]}")
    
    log.append(f"[PASS2] Discovered {len(candidate_chunks)} total candidate chunks")
    
    # Step 4: Determine result
    success = len(candidate_chunks) > 0
    reason = "pass2_success" if success else "no_intent_aligned_evidence"
    
    return {
        "success": success,
        "expanded_queries": expanded_queries,
        "candidate_chunks": candidate_chunks,
        "candidate_count": len(candidate_chunks),
        "restricted_pages": restricted_pages,
        "pass2_reason": reason,
        "log": log
    }


def _should_abstain(chunks: list, question_intent: str, graph_validated: bool, match_quality: str = "weak") -> tuple[bool, str]:
    """
    STRICT INTENT-AWARE ABSTENTION with hard rules.
    Returns: (should_abstain: bool, reason: str)
    
    Rules for intent enforcement:
    - Definition/Procedure/Comparison: HARD ABSTAIN on weak graph validation or evidence mismatch
    - Fact_lookup/Explanation: More lenient, but still abstain if no evidence matches
    - YesNo: Allow abstention if no definitive answer
    """
    # HARD RULE 1: No graph validation = MUST abstain
    if not graph_validated:
        return True, "Concepts not found in knowledge base. Cannot answer without source validation."
    
    # HARD RULE 2: Weak validation for strict intents = MUST abstain
    if match_quality == "weak" and question_intent in ["definition", "procedure", "comparison"]:
        return True, f"Only partial concept match. Cannot reliably provide {question_intent}. Abstaining."
    
    # HARD RULE 3: No chunks = MUST abstain
    if not chunks:
        return True, "No content found after retrieval and filtering."
    
    # HARD RULE 4: Low quality chunks
    avg_length = sum(len(c.get("content", "").split()) for c in chunks) / len(chunks)
    if avg_length < 50:
        return True, "Content too brief to answer reliably."
    
    # HARD RULE 5: MANDATORY evidence-intent matching
    matching, rejected = _filter_by_evidence_type(chunks, question_intent)
    if not matching:
        print(f"[HARD ABSTAIN] [X] INTENT={question_intent}: ZERO chunks have {question_intent} evidence")
        print(f"[HARD ABSTAIN] Rejected {len(rejected)} chunks")
        return True, f"No {question_intent} evidence found. Cannot answer without proper source material."
    
    # HARD RULE 6: For definitions and procedures, require at least 1 matching chunk
    if question_intent in ["definition", "procedure"]:
        match_pct = (len(matching) / len(chunks)) * 100
        if len(matching) == 0:
            print(f"[HARD ABSTAIN] [X] INTENT={question_intent}: ZERO chunks qualify ({match_pct:.0f}%)")
            return True, f"Insufficient {question_intent} evidence (need >0%, got {match_pct:.0f}%). Abstaining."
        else:
            print(f"[HARD ABSTAIN] [OK] INTENT={question_intent}: {len(matching)} chunks qualify ({match_pct:.0f}%)")
    
    print(f"[ABSTENTION CHECK] [OK] PASS: Can answer {question_intent} with {len(matching)} valid chunks")
    return False, ""


def _apply_final_gates(chunks: list, question_intent: str, core_concepts: list, entities: list) -> tuple[bool, str]:
    """
    UNIFIED FINAL GATES - Applied after both Pass1 and Pass2.
    Enforces: intent alignment, concept relevance, concept completeness.
    
    Returns: (pass_gates: bool, failure_reason: str)
    """
    if not chunks:
        return False, "no_chunks_after_retrieval"
    
    # GATE 1: Evidence-Intent Alignment (MANDATORY)
    matching, rejected = _filter_by_evidence_type(chunks, question_intent)
    if not matching:
        print(f"[FINAL GATES] REJECT: No {question_intent} evidence found")
        return False, "no_intent_aligned_evidence"
    
    # GATE 2: Concept Relevance Threshold (definition-specific, token-level overlap)
    if question_intent == "definition":
        # Use token-level overlap (more flexible than phrase matching)
        matched_content = "\n".join(c.get("content", "") for c in matching)
        matched_content_lower = matched_content.lower()
        
        # Count tokens from core concepts that appear
        tokens_found = set()
        for concept in core_concepts:
            tokens = concept.lower().split()
            for token in tokens:
                if token in matched_content_lower:
                    tokens_found.add(token)
        
        # Use all core concept words as pool
        all_tokens = set()
        for concept in core_concepts:
            all_tokens.update(concept.lower().split())
        
        concept_relevance = (len(tokens_found) / max(1, len(all_tokens))) if all_tokens else 1.0
        print(f"[FINAL GATES] Token-level concept relevance: {len(tokens_found)}/{len(all_tokens)} = {concept_relevance:.2f}")
        
        # Require at least configured threshold coverage
        threshold = DEFINITION_CONCEPT_THRESHOLD
        if concept_relevance < threshold:
            print(f"[FINAL GATES] REJECT: Low concept relevance ({concept_relevance:.2f} < {threshold:.2f})")
            return False, "insufficient_concept_coverage"
    
    # GATE 3: Concept Completeness (at least one full concept phrase should be present)
    if question_intent == "definition":
        matched_content = " ".join(c.get("content", "").lower() for c in matching)
        
        # Check if any multi-word concept appears
        multi_word_concepts = [c for c in core_concepts if len(c.split()) > 1]
        complete_match_found = False
        
        for concept in multi_word_concepts:
            if concept.lower() in matched_content:
                complete_match_found = True
                print(f"[FINAL GATES] Found complete concept match: '{concept}'")
                break
        
        if not complete_match_found and multi_word_concepts:
            print(f"[FINAL GATES] REJECT: No complete concept phrase found")
            return False, "no_complete_concept_match"
    
    print(f"[FINAL GATES] PASS: All gates passed for {question_intent}")
    return True, ""


def _semantic_filter(query: str, context_texts, core_concepts=None, entities=None):
    """
    Post-retrieval semantic validation to catch concept drift.
    Uses HARD CONSTRAINTS from core concepts to filter chunks.
    Filters out chunks that match lexically but not semantically.
    Also rejects index/glossary pages.
    
    Args:
        query: The user's question
        context_texts: Retrieved chunks
        core_concepts: List of core concept phrases/terms (HARD CONSTRAINTS)
        entities: List of domain entities
    """
    query_lower = query.lower()
    core_concepts = core_concepts or []
    entities = entities or []

    # Lightweight question-type detection and concept extraction (definition-focused)
    STOPWORDS = {"what", "is", "are", "the", "a", "an", "of", "define", "definition", "please", "tell", "me", "about", "explain", "meaning"}

    def _detect_qtype_and_concept(q: str):
        qtype = None
        concept_tokens = set()
        prefix_hits = ["what is", "what are", "definition of", "define", "meaning of", "what does"]
        for p in prefix_hits:
            if p in q:
                qtype = "definition"
                after = q.split(p, 1)[1].strip()
                concept_tokens = {t for t in _tokenize(after) if t not in STOPWORDS}
                break
        return qtype, concept_tokens

    question_type, concept_tokens = _detect_qtype_and_concept(query_lower)

    # Intent detection for section-aware filtering
    def _detect_intent(q: str):
        if "biosafety" in q or "bsl" in q or "containment" in q:
            return "biosafety"
        if "ppe" in q or "protective equipment" in q:
            return "ppe"
        return None

    intent = _detect_intent(query_lower)

    # Section rules (soft): prefer chapter-range and keyword hits
    SECTION_RULES = {
        "biosafety": {
            "page_range": (70, 260),  # soft window; keep if inside or keyword hit
            "keywords": ["biosafety", "bsl", "containment", "risk group", "risk assessment"],
        },
        "ppe": {
            "page_range": (20, 180),
            "keywords": ["ppe", "protective equipment", "gown", "mask", "glove", "respirator"],
        },
    }
    
    # First: Detect and remove index/glossary pages (they contain lots of numbers and cross-refs)
    filtered = []
    for ctx in context_texts:
        content = ctx.get("content", "").lower()
        
        # Count characteristics of index pages
        has_many_numbers = len([c for c in content if c.isdigit()]) > len(content) * 0.15  # > 15% digits
        has_page_refs = " " in content and content.count(",") > content.count(".") * 2  # More commas than periods
        has_cross_refs = "–" in content or "û" in content  # Unicode dashes used in indices
        looks_like_index = has_many_numbers and has_page_refs and has_cross_refs
        
        if looks_like_index:
            print(f"[FILTER] Rejected index page: {ctx.get('id')}")
            continue
        
        # Check minimum content length
        if len(content.split()) < 15:
            print(f"[FILTER] Rejected too short: {ctx.get('id')}")
            continue
        
        filtered.append(ctx)
    
    if not filtered:
        return [], True  # No valid context found, filter was applied
    
    # Define semantic rules: query intent → required concepts → excluded concepts
    semantic_rules = {
        "ppe": {
            "triggers": ["personal protective equipment", "ppe", "protective equipment", "protection equipment", "protective gear"],
            "required_concepts": ["glove", "gown", "coat", "mask", "goggle", "face shield", "respirator", "apron", "shoe cover", "head cover", "eye protection"],
            "excluded_concepts": ["microscope", "centrifuge", "balance", "refrigerator", "incubator", "autoclave", "pipette", "freezer", "shaker", "analyzer"]
        },
        "biosafety_levels": {
            "triggers": ["biosafety level", "bsl", "containment level"],
            "required_concepts": ["level 1", "level 2", "level 3", "level 4", "bsl", "containment"],
            "excluded_concepts": []
        }
    }
    
    # Check if query matches any semantic rule
    active_rule = None
    rule_name_matched = None
    for rule_name, rule in semantic_rules.items():
        if any(trigger in query_lower for trigger in rule["triggers"]):
            active_rule = rule
            rule_name_matched = rule_name
            break
    
    if not active_rule:
        return context_texts, False  # No rule applies, return all + no filtering
    
    # Filter chunks based on semantic relevance (RELAXED - keep more context)
    filtered = []
    for chunk in context_texts:
        content_lower = chunk.get("content", "").lower()

        # SOFT FILTERING: Only reject if has MANY excluded concepts AND NO required concepts
        has_excluded = sum(1 for exc in active_rule["excluded_concepts"] if exc in content_lower)
        has_required = sum(1 for req in active_rule["required_concepts"] if req in content_lower)

        # Keep chunk if it has required concepts OR has few excluded concepts
        if has_required >= 1 or has_excluded <= 1:
            filtered.append(chunk)  # Keep it - worth trying

    # Section-aware filter: prefer chunks in relevant chapter/page window or with keywords
    def _section_filter(chunks):
        if not intent or intent not in SECTION_RULES:
            return chunks, False
        rule = SECTION_RULES[intent]
        pr = rule.get("page_range")
        kw = rule.get("keywords", [])
        keep = []
        for c in chunks:
            page = c.get("page")
            content_lower = c.get("content", "").lower()
            kw_hit = any(k in content_lower for k in kw)
            page_hit = False
            try:
                if page is not None and pr:
                    page_num = int(page)
                    page_hit = pr[0] <= page_num <= pr[1]
            except Exception:
                page_hit = False
            if kw_hit or page_hit:
                keep.append(c)
        return (keep, True) if keep else (chunks, False)

    filtered, section_applied = _section_filter(filtered)

    # Graph-based entity validation: require at least one entity neighbor matching intent keywords
    def _graph_entity_validate(chunks):
        if not intent:
            return chunks, False
        required = {
            "biosafety": ["biosafety", "bsl", "containment"],
            "ppe": ["ppe", "protective equipment", "glove", "mask", "respirator"],
        }.get(intent, [])
        if not required:
            return chunks, False

        validated = []
        for c in chunks:
            chunk_ok = False
            content_lower = c.get("content", "").lower()
            if any(r in content_lower for r in required):
                chunk_ok = True
            if not chunk_ok:
                try:
                    neigh = GRAPH.neighbors(c.get("id"), limit=30, rel_types=["HAS_ENTITY", "IN_SECTION", "SIMILAR_TO"])
                    for n in neigh:
                        labels = [str(lbl).lower() for lbl in n.get("labels", [])]
                        nid = str(n.get("id", "")).lower()
                        if any(r in labels or r in nid for r in required):
                            chunk_ok = True
                            break
                except Exception as e:
                    print(f"[WARN] Graph entity validation skipped for {c.get('id')}: {e}")
            if chunk_ok:
                validated.append(c)

        return (validated, True) if validated else (chunks, False)

    filtered, graph_applied = _graph_entity_validate(filtered)

    # Concept-level constraint: definition questions must keep chunks mentioning the concept (or its graph neighbors)
    def _concept_filter(chunks):
        if question_type != "definition" or not concept_tokens:
            return chunks, False
        concept_token_expanded = set()
        for ct in concept_tokens:
            concept_token_expanded.update(_normalized_tokens(ct))
        keep = []
        for c in chunks:
            content_tokens = _normalized_tokens(c.get("content", ""))
            overlap = len(content_tokens & concept_token_expanded)
            needed = max(1, len(concept_token_expanded) // 2)
            if overlap >= needed:
                keep.append(c)
                continue
            try:
                neigh = GRAPH.neighbors(c.get("id"), limit=20, rel_types=["HAS_ENTITY", "SIMILAR_TO", "IN_SECTION"])
                matched = False
                for n in neigh:
                    labels = [str(lbl).lower() for lbl in n.get("labels", [])]
                    nid = str(n.get("id", "")).lower()
                    neighbor_tokens = _normalized_tokens(" ".join(labels + [nid]))
                    if concept_token_expanded & neighbor_tokens:
                        matched = True
                        break
                if matched:
                    keep.append(c)
            except Exception as e:
                print(f"[WARN] Concept filter skipped for {c.get('id')}: {e}")
        return (keep, True) if keep else (chunks, False)

    filtered, concept_applied = _concept_filter(filtered)
    
    # HARD CONSTRAINT: Core concepts from question MUST appear in chunks
    def _hard_concept_constraint(chunks):
        """Enforce core concepts as HARD constraints - chunks MUST contain at least one"""
        if not core_concepts:
            return chunks, False
        
        print(f"[HARD CONSTRAINT] Filtering {len(chunks)} chunks by core concepts: {core_concepts[:3]}")
        
        keep = []
        for c in chunks:
            content_tokens = _normalized_tokens(c.get("content", ""))

            has_concept = False
            matched_concepts = []

            for concept in core_concepts:
                score = _concept_match_score(concept, content_tokens)
                if score >= 0.5:
                    has_concept = True
                    matched_concepts.append(f"{concept} ({score:.2f})")

            if not has_concept and entities:
                for entity in entities:
                    score = _concept_match_score(entity, content_tokens)
                    if score >= 0.5:
                        has_concept = True
                        matched_concepts.append(f"{entity} ({score:.2f})")
                        break

            if has_concept:
                keep.append(c)
                print(f"[HARD CONSTRAINT] [OK] Kept {c.get('id')} - matched: {matched_concepts[:2]}")
            else:
                print(f"[HARD CONSTRAINT] [X] Rejected {c.get('id')} - no concept match")
        
        if not keep:
            print(f"[HARD CONSTRAINT] WARNING: All chunks rejected! Falling back to original set.")
            return chunks, False  # Fallback to avoid empty results
        
        return (keep, True)
    
    filtered, hard_constraint_applied = _hard_concept_constraint(filtered)

    # Return filtered results + whether filtering happened
    filter_was_applied = len(filtered) != len(context_texts) or section_applied or graph_applied or concept_applied or hard_constraint_applied
    return (filtered, filter_was_applied)


def _build_prompt(context_texts, question):
    if not context_texts:
        return f"No relevant context found for: {question}"
    
    header = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If you cannot answer from the context, say you don't know.\n\n"
    )
    ctx = []
    for i, t in enumerate(context_texts, 1):
        ctx.append(f"[Text {i}] (ID: {t['id']}, Page: {t.get('page', 'N/A')}):\n{t.get('content','')}\n")
    return header + "\n".join(ctx) + f"\nQuestion: {question}\nAnswer:"


def _llm_generate(prompt, context_texts=None):
    # Check if no context was provided (semantic filter rejected everything)
    if "No relevant context found" in prompt:
        return "No semantically relevant context found in the retrieved documents for this question."
    
    # If we already have structured context, use it directly; otherwise parse the prompt
    ctx = context_texts if context_texts is not None else []
    if context_texts is None:
        lines = prompt.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if '[Text ' in line and '(ID:' in line and ']:' in line:
                i += 1
                chunk_text = []
                while i < len(lines):
                    if lines[i].strip() and not lines[i].startswith('[Text ') and not lines[i].startswith('Question:'):
                        chunk_text.append(lines[i].strip())
                        i += 1
                    else:
                        break
                if chunk_text:
                    ctx.append(' '.join(chunk_text))
            else:
                i += 1
    
    if ctx:
        # Normalize to plain text strings if dicts were provided
        norm_ctx = []
        for item in ctx:
            if isinstance(item, dict):
                norm_ctx.append(item.get("content") or item.get("text") or "")
            else:
                norm_ctx.append(str(item))

        # Prefer sentences that mention storage/handling keywords, fallback to first sentences
        keywords = {"refrigerator", "reagent", "reagents", "pregnancy", "keep", "storage", "store"}
        sentences = []
        for chunk in norm_ctx:
            for sent in re.split(r'[.!?]+\s+', chunk):
                clean = sent.strip()
                if len(clean) <= 10:
                    continue
                tokens = set(_tokenize(clean))
                if tokens & keywords:
                    sentences.append(clean + '.')
                if len(sentences) >= 3:
                    break
            if len(sentences) >= 3:
                break

        if not sentences:
            # If no keyword hit, fallback to the first 2 sentences from the first chunk
            fallback_sentences = []
            for sent in re.split(r'[.!?]+\s+', norm_ctx[0]):
                clean = sent.strip()
                if len(clean) > 10:
                    fallback_sentences.append(clean + '.')
                if len(fallback_sentences) >= 2:
                    break
            sentences = fallback_sentences

        return ' '.join(sentences) if sentences else norm_ctx[0][:300]
    
    return "No relevant information found in the provided context."


def _tokenize(text: str):
    return [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(w) > 2]


def _normalized_tokens(text: str) -> set[str]:
    """Normalize text into tokens and expand with lightweight synonyms."""
    base_tokens = set(_tokenize(text))
    expanded = set(base_tokens)
    for tok in list(base_tokens):
        for syn in SYNONYM_MAP.get(tok, []):
            expanded.update(_tokenize(syn))
    return expanded


def _concept_match_score(concept: str, target_tokens: set[str]) -> float:
    """Compute token overlap score between a concept phrase and target tokens."""
    concept_tokens = _normalized_tokens(concept)
    if not concept_tokens:
        return 0.0
    return len(concept_tokens & target_tokens) / len(concept_tokens)


def _generate_llm_answer(question: str, context_texts: list, question_intent: str = "fact_lookup", max_length=150):
    """
    Generate answer using LLM with STRICT synthesis-only constraints.
    LLM may ONLY: summarize, rephrase, structure existing content
    LLM may NOT: introduce new facts, infer without evidence, answer without citations
    
    Returns: (answer_text, confidence_score)
    """
    if not LLM_READY:
        # Fallback to extraction if LLM not available
        return (_llm_generate(None, context_texts=context_texts), 0.5)
    
    # Build prompt with STRICT synthesis constraints and citation requirements
    context_str = "\n\n".join([
        f"[Source {i+1} - ID:{ctx.get('id')} Page:{ctx.get('page')}]:\n{ctx.get('content', '')}" 
        for i, ctx in enumerate(context_texts[:8])
    ])
    
    prompt = f"""You are a synthesis assistant. Your role is STRICTLY LIMITED to:
[OK] ALLOWED: Summarize, rephrase, and structure information from the provided sources
[X] FORBIDDEN: Introduce new facts, infer definitions without direct evidence, make assumptions

CRITICAL RULES:
1. Answer ONLY using explicit information from the sources below
2. DO NOT add facts not present in the sources
3. DO NOT infer or deduce beyond what is explicitly stated
4. If information is not in the sources, respond: "Not found in the provided sources"
5. Include source references [Source X] for key statements

Sources:
{context_str}

Question: {question}

Answer (with source citations):Context:
{context_str}

Question: {question}

Answer:"""
    
    try:
        # Generate answer
        result = LLM_PIPE(prompt, max_length=max_length, do_sample=False)
        answer = result[0]['generated_text'].strip()
        
        # Calculate confidence based on overlap with context
        answer_tokens = set(_tokenize(answer))
        context_tokens = set()
        for ctx in context_texts[:5]:
            context_tokens.update(_tokenize(ctx.get('content', '')))
        
        if len(answer_tokens) == 0:
            confidence = 0.0
        else:
            overlap = len(answer_tokens & context_tokens)
            confidence = min(1.0, overlap / len(answer_tokens))
        
        # Boost confidence if answer is substantial
        if len(answer.split()) >= 5:
            confidence = min(1.0, confidence + 0.2)
        
        # INTENT-AWARE CONFIDENCE ADJUSTMENT
        # If definition intent, check for definitional language
        if question_intent == "definition":
            has_definition_language = bool(re.search(r'\b(is|are|refers?\s+to|defined\s+as|means?|consists?\s+of)\b', answer, re.I))
            if not has_definition_language:
                # Penalize non-definitional answers to definition questions
                confidence = max(0.0, confidence - 0.3)
                print(f"[CONFIDENCE] Penalized for missing definition language: {confidence:.2f}")
        
        # CRITICAL: Reject low-confidence answers to reduce hallucination
        if confidence < 0.3 and "don't" not in answer.lower() and "i don't" not in answer.lower():
            return (f"Insufficient confidence in answer. Retrieved context may not contain accurate information.", 0.2)
        
        return (answer, confidence)
    
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return (_llm_generate(None, context_texts=context_texts), 0.5)


def _metrics(answer: str, context_texts, core_concepts=None, question_intent=None):
    """
    NEW EVALUATION CRITERIA (replacing coverage/overlap):
    1. Concept relevance - Are core concepts mentioned in the answer?
    2. Evidence-intent alignment - Does answer match question type?
    3. Citation correctness - Are sources cited properly?
    4. Abstention correctness - Did we abstain when appropriate?
    """
    metrics = {}
    
    # 1. Concept Relevance - Check if core concepts appear in answer
    if core_concepts:
        answer_lower = answer.lower()
        concepts_found = sum(1 for c in core_concepts if c.lower() in answer_lower)
        concept_relevance = (concepts_found / max(1, len(core_concepts))) * 100.0
        metrics["concept_relevance_percent"] = round(concept_relevance, 1)
    else:
        metrics["concept_relevance_percent"] = 0.0
    
    # 2. Evidence-Intent Alignment - Does answer match question type?
    intent_aligned = False
    if question_intent == "definition":
        # Should contain definitional language: "is", "refers to", "defined as"
        intent_aligned = bool(re.search(r'\b(is|are|refers?\s+to|defined\s+as|means?|consists?\s+of)\b', answer, re.I))
    elif question_intent == "procedure":
        # Should contain procedural language: steps, process, numbered lists
        intent_aligned = bool(re.search(r'\b(step|process|procedure|first|then|next|finally|\d+\.)\b', answer, re.I))
    elif question_intent == "comparison":
        # Should contain comparative language: difference, versus, compared to
        intent_aligned = bool(re.search(r'\b(difference|differ|versus|vs|compared\s+to|contrast|while|whereas)\b', answer, re.I))
    elif question_intent in ["fact_lookup", "explanation", "yesno"]:
        # Should contain factual language from sources
        intent_aligned = len(answer.split()) >= 5  # At least substantive answer
    
    metrics["evidence_intent_aligned"] = intent_aligned
    
    # 3. Citation Correctness - Check for [Source X] citations
    citations = re.findall(r'\[Source\s+\d+[^\]]*\]', answer)
    has_citations = len(citations) > 0
    citation_count = len(citations)
    
    # Verify citations match actual context sources
    valid_citations = 0
    for i, ctx in enumerate(context_texts[:10], 1):
        if f"[Source {i}" in answer:
            valid_citations += 1
    
    metrics["citation_count"] = citation_count
    metrics["valid_citations"] = valid_citations
    metrics["has_citations"] = has_citations
    
    # 4. Abstention Correctness - Check if abstention was appropriate
    abstained = "cannot answer" in answer.lower() or "insufficient" in answer.lower()
    metrics["abstained"] = abstained
    
    return metrics


@app.get("/")
def root():
    return {
        "message": "Simple RAG API - Just ask your question!",
        "endpoints": {
            "/ask": "POST your question here",
            "/docs": "Interactive API documentation"
        }
    }


@app.post("/ask")
def ask(req: QuestionRequest):
    """
    Simple endpoint: Just send your question, get compared answers.
    
    The system will:
    0. Detect question intent BEFORE retrieval
    1. Search with FAISS only (vector search)
    2. Search with FAISS + Neo4j graph (hybrid)
    3. Generate answers from both methods
    4. Compare which method is better
    """
    # STEP 0: Detect question intent BEFORE retrieval (REQUIRED)
    intent_info = _detect_question_intent(req.query)
    question_intent = intent_info["intent"]
    core_concepts = intent_info["core_concepts"]
    entities = intent_info["entities"]
    
    print(f"[INTENT] Detected: {question_intent}")
    print(f"[CORE CONCEPTS] Hard constraints: {core_concepts[:5]}")
    print(f"[ENTITIES] Domain entities: {entities}")
    print(f"[ENFORCEMENT] Will enforce strict intent-based validation")
    
    # STEP 0.5: GRAPH-FIRST VALIDATION - Query Neo4j BEFORE retrieval
    print(f"\n[GRAPH-FIRST] Validating concepts in knowledge graph...")
    graph_validation = GRAPH.validate_concepts(core_concepts, entities)
    
    if not graph_validation["exists"]:
        print(f"[GRAPH-FIRST] [X] RULE 1 TRIGGERED: No graph nodes found. Will abstain if no other validation.")
        allowed_text_ids = None
        allowed_semantic_ids = None
        validated_pages = None
        content_roles = None
    else:
        match_quality = graph_validation.get("match_quality", "weak")
        fallback_triggered = graph_validation.get("fallback_triggered", False)
        fallback_status = " [FALLBACK]" if fallback_triggered else ""
        
        print(f"[GRAPH-FIRST] [OK] VALIDATED: Quality={match_quality}{fallback_status}")
        if match_quality == "phrase":
            print(f"[GRAPH-FIRST] [OK] Strong phrase match (canonical concepts found)")
        elif match_quality == "alias":
            print(f"[GRAPH-FIRST] [!] Alias match (using synonym variants, fallback triggered)")
        elif match_quality == "section":
            print(f"[GRAPH-FIRST] [!] Section fallback (using biosafety pages, will combine with FAISS)")
        
        allowed_text_ids = graph_validation["text_ids"]
        validated_pages = graph_validation["pages"]
        content_roles = graph_validation["roles"]
        matched_concepts = graph_validation["matched_concepts"]
        
        print(f"[GRAPH-FIRST] [OK] Found {len(allowed_text_ids)} validated chunks")
        print(f"[GRAPH-FIRST] Matched concepts: {matched_concepts[:5]}")
        print(f"[GRAPH-FIRST] Pages: {validated_pages[:10]}")
        print(f"[GRAPH-FIRST] Content roles: {set(content_roles)}")
        
        # Map text IDs to semantic IDs for FAISS filtering
        reverse_mapping = {v: k for k, v in SEMANTIC_TO_NEO4J.items()}
        allowed_semantic_ids = set(reverse_mapping.get(tid) for tid in allowed_text_ids if reverse_mapping.get(tid))
        print(f"[GRAPH-FIRST] Allowed semantic IDs for FAISS: {len(allowed_semantic_ids)}")
    
    index_path, meta_path, chunks_path = _paths_for_search()
    if not index_path or not meta_path or not chunks_path:
        raise HTTPException(status_code=500, detail="Search index not found on server.")

    # Adjust retrieval parameters based on intent
    INTENT_PARAMS = {
        "definition": {"faiss_topk": 15, "semantic_weight": 0.7, "bm25_weight": 0.3, "alpha": 0.8, "beta": 0.2},
        "explanation": {"faiss_topk": 20, "semantic_weight": 0.6, "bm25_weight": 0.4, "alpha": 0.7, "beta": 0.3},
        "comparison": {"faiss_topk": 25, "semantic_weight": 0.5, "bm25_weight": 0.5, "alpha": 0.6, "beta": 0.4},
        "procedure": {"faiss_topk": 20, "semantic_weight": 0.4, "bm25_weight": 0.6, "alpha": 0.7, "beta": 0.3},
        "fact_lookup": {"faiss_topk": 12, "semantic_weight": 0.5, "bm25_weight": 0.5, "alpha": 0.7, "beta": 0.3},
        "yesno": {"faiss_topk": 10, "semantic_weight": 0.6, "bm25_weight": 0.4, "alpha": 0.8, "beta": 0.2},
    }
    
    params = INTENT_PARAMS.get(question_intent, INTENT_PARAMS["fact_lookup"])
    FAISS_TOPK = params["faiss_topk"]
    SEMANTIC_WEIGHT = params["semantic_weight"]
    BM25_WEIGHT = params["bm25_weight"]
    ALPHA = params["alpha"]
    BETA = params["beta"]
    GRAPH_NEIGH_LIMIT = 15
    TOP_N_CONTEXT = 10

    try:
        # 1. Run FAISS search RESTRICTED to graph-validated sections only
        print(f"\n[FAISS] Running search with graph constraints...")
        faiss_hits = _faiss_only_search(
            req.query, 
            FAISS_TOPK * 2,  # Fetch more to compensate for filtering
            Path(index_path), 
            Path(meta_path),
            allowed_ids=allowed_semantic_ids  # RESTRICT to graph-validated IDs
        )
        print(f"[FAISS] Got {len(faiss_hits)} graph-validated hits (semantic_id, score): {faiss_hits[:5]}")
        
        # 2. Map to Neo4j text IDs and fetch chunks
        faiss_ids = _map_semantic_to_text_ids(faiss_hits)
        print(f"[FAISS] Mapped to Neo4j IDs: {faiss_ids[:5]}")
        
        faiss_texts_all = _fetch_texts_by_ids(faiss_ids)
        print(f"[FAISS] Fetched {len(faiss_texts_all)} text chunks")
        for i, chunk in enumerate(faiss_texts_all[:3]):
            print(f"[FAISS] Chunk {i}: ID={chunk.get('id')}, Page={chunk.get('page')}, Text[:100]={chunk.get('content', '')[:100]}")
        
        faiss_texts_raw = faiss_texts_all
        
        # 3. Apply semantic filtering with HARD CONSTRAINTS from core concepts
        faiss_texts, faiss_filtered = _semantic_filter(req.query, faiss_texts_raw, core_concepts=core_concepts, entities=entities)
        
        # 4. EVIDENCE TYPE MATCHING - MANDATORY check: chunks MUST match question intent
        print(f"\n[EVIDENCE MATCHING] RULE 5: Enforcing mandatory evidence-intent matching for {question_intent}")
        evidence_matched, evidence_rejected = _filter_by_evidence_type(faiss_texts, question_intent)
        print(f"[EVIDENCE MATCHING] Result: {len(evidence_matched)} passed, {len(evidence_rejected)} rejected")
        if len(evidence_matched) == 0:
            print(f"[EVIDENCE MATCHING] [X] ZERO chunks have {question_intent} evidence - will trigger hard abstention")

        # 4.5 STRICT POST-EVIDENCE LOCK for definitions: enforce concept completeness
        if question_intent == "definition":
            # Consider only multi-word core concepts for strict matching
            multi_word_core = []
            for c in core_concepts:
                if (" " in c) or (len(c) > 5):
                    multi_word_core.append(c.lower())
            multi_word_core = list(dict.fromkeys(multi_word_core))[:5]  # dedupe and cap

            joined = " \n".join([ch.get("content", "") for ch in evidence_matched[:TOP_N_CONTEXT]])
            evidence_tokens = _normalized_tokens(joined)

            cover_scores = []
            for c in multi_word_core:
                score = _concept_match_score(c, evidence_tokens)
                cover_scores.append(score)

            total = max(1, len(cover_scores))
            matched = sum(1 for s in cover_scores if s >= DEFINITION_CONCEPT_THRESHOLD)
            concept_relevance = matched / total
            print(f"[DEFINITION LOCK] Coverage: {matched}/{total} concepts met (token overlap >= {DEFINITION_CONCEPT_THRESHOLD:.2f}); max score={max(cover_scores or [0]):.2f}")
            
            # When fallback is triggered, be more lenient on definition lock
            fallback_triggered = graph_validation.get("fallback_triggered", False)
            fallback_status = " [FALLBACK_MODE: lenient]" if fallback_triggered else ""
            
            # Mandatory abstention for definitions when evidence not aligned OR concept incomplete
            # Exception: if using section fallback, we relax the threshold slightly
            threshold_to_use = 0.4 if fallback_triggered else DEFINITION_CONCEPT_THRESHOLD
            if (len(evidence_matched) == 0) or (concept_relevance < threshold_to_use):
                reason = ""
                if len(evidence_matched) == 0:
                    reason = "No definition evidence found in retrieved documents."
                elif concept_relevance < threshold_to_use:
                    reason = f"Concept mention insufficient (threshold {threshold_to_use:.2f}, got {concept_relevance:.2f}){fallback_status}"
                print(f"[HARD ABSTAIN] [X] Definition lock: {reason}")
                return {
                    "question": req.query,
                    "answer": "Not found in the provided documents.",
                    "abstained": True,
                    "reason": reason,
                    "confidence": 0.0,
                    "intent": question_intent,
                    "core_concepts": core_concepts[:5],
                    "graph_validation": {
                        "exists": graph_validation["exists"],
                        "matched_concepts": graph_validation.get("matched_concepts", [])[:5],
                        "validated_chunks": len(allowed_text_ids) if allowed_text_ids else 0,
                        "match_quality": graph_validation.get("match_quality", "none"),
                        "fallback_triggered": fallback_triggered,
                    },
                    "evidence_matching": {
                        "matched": len(evidence_matched),
                        "rejected": len(evidence_rejected),
                        "rejection_reasons": [r for _, r in evidence_rejected[:5]]
                    }
                }

        # 5. ABSTENTION LOGIC - Decide if we should answer or abstain
        should_abstain, abstention_reason = _should_abstain(
            evidence_matched, 
            question_intent, 
            graph_validated=graph_validation["exists"],
            match_quality=graph_validation.get("match_quality", "weak")  # Signal strength of graph validation
        )
        
        pass2_result = None
        pass2_used = False
        pass2_gates_failed = None
        
        if should_abstain:
            # STAGE 4: Attempt PASS 2 soft re-discovery (once only)
            print(f"\n[PASS 1] Insufficient evidence ({len(evidence_matched)} chunks). Triggering PASS 2...")
            print(f"[ABSTENTION] {abstention_reason}")
            
            try:
                pass2_result = _run_pass2_discovery(
                    query=req.query,
                    core_concepts=core_concepts,
                    entities=entities,
                    graph_validation=graph_validation,
                    index_path=index_path,
                    meta_path=meta_path,
                    chunks_path=chunks_path,
                    restricted_pages=validated_pages
                )
                
                # Log Pass 2 results
                for log_msg in pass2_result.get("log", []):
                    print(log_msg)
                
                if pass2_result["success"] and pass2_result["candidate_chunks"]:
                    print(f"\n[PASS 2] Re-discovery successful! Found {pass2_result['candidate_count']} candidate chunks")
                    
                    # Re-apply validation to Pass 2 candidates (SAME strict rules)
                    pass2_semantic, pass2_filtered = _semantic_filter(
                        req.query, 
                        pass2_result["candidate_chunks"][:TOP_N_CONTEXT],
                        core_concepts=core_concepts,
                        entities=entities
                    )
                    
                    pass2_evidence, pass2_rejected = _filter_by_evidence_type(pass2_semantic, question_intent)
                    print(f"[PASS 2 EVIDENCE] {len(pass2_evidence)} chunks match intent (after validation)")
                    
                    if pass2_evidence:
                        # CRITICAL: Apply FINAL GATES to Pass 2 results (same as Pass1)
                        print(f"\n[PASS 2] Applying final gates (intent alignment, concept relevance, completeness)...")
                        gates_pass, gates_reason = _apply_final_gates(
                            pass2_evidence,
                            question_intent,
                            core_concepts,
                            entities
                        )
                        
                        if gates_pass:
                            # Use Pass 2 evidence!
                            evidence_matched = pass2_evidence
                            faiss_texts = pass2_semantic
                            pass2_used = True
                            should_abstain = False
                            print(f"[PASS 2 SUCCESS] Using {len(evidence_matched)} validated chunks from Pass 2")
                        else:
                            # Pass 2 gates failed
                            pass2_gates_failed = gates_reason
                            pass2_result["pass2_reason"] = gates_reason
                            print(f"[PASS 2 FAILED] Final gates rejected: {gates_reason}")
                    else:
                        # Pass 2 found candidates but they don't match intent
                        pass2_result["pass2_reason"] = "no_intent_aligned_evidence"
                        print(f"[PASS 2 FAILED] Found {pass2_result['candidate_count']} chunks but none match {question_intent} intent")
                else:
                    # Pass 2 found nothing
                    if not pass2_result["success"]:
                        print(f"[PASS 2 FAILED] Reason: {pass2_result['pass2_reason']}")
            except Exception as e:
                print(f"[PASS 2] Exception during re-discovery: {str(e)[:100]}")
                pass2_result = {"pass2_reason": "pass2_exception", "log": [str(e)[:100]]}
        
        if should_abstain:
            # Still need to abstain after Pass 1 and Pass 2 both failed
            precise_reason = "no_intent_aligned_evidence"
            if pass2_gates_failed:
                # Pass 2 gates explicitly failed
                precise_reason = pass2_gates_failed
            elif pass2_result:
                precise_reason = pass2_result.get("pass2_reason", "unknown")
            
            print(f"[ABSTAIN FINAL] After Pass 1 and Pass 2: {precise_reason}")
            return {
                "question": req.query,
                "answer": "Not found in the provided documents.",
                "abstained": True,
                "reason": abstention_reason,
                "pass2_attempted": True,
                "pass2_reason": precise_reason,
                "confidence": 0.0,
                "intent": question_intent,
                "core_concepts": core_concepts[:5],
                "graph_validation": {
                    "exists": graph_validation["exists"],
                    "matched_concepts": graph_validation.get("matched_concepts", [])[:5],
                    "validated_chunks": len(allowed_text_ids) if allowed_text_ids else 0,
                    "match_quality": graph_validation.get("match_quality", "none"),
                    "fallback_triggered": graph_validation.get("fallback_triggered", False),
                },
                "evidence_matching": {
                    "matched": len(evidence_matched),
                    "rejected": len(evidence_rejected),
                    "rejection_reasons": [r for _, r in evidence_rejected[:5]]
                },
                "pass2": {
                    "attempted": True,
                    "expanded_queries": pass2_result.get("expanded_queries", [])[:5] if pass2_result else [],
                    "restricted_pages": pass2_result.get("restricted_pages", [])[:10] if pass2_result else [],
                    "candidates_found": pass2_result.get("candidate_count", 0) if pass2_result else 0,
                    "reason": precise_reason,
                    "gates_failed": pass2_gates_failed
                }
            }
        
        # Use evidence-matched chunks for answer generation
        final_chunks = evidence_matched if evidence_matched else faiss_texts
        print(f"[FINAL] Using {len(final_chunks)} chunks for answer generation")
        
        # 5.5 STAGE 3: EVIDENCE BUNDLING AND RERANKING (for ALL intents, especially definitions)
        # APPLY BUNDLING EVEN FOR PASS 2 RESULTS (no bypass)
        if len(final_chunks) > 0:
            print(f"\n[BUNDLING] Starting evidence bundling for {question_intent} question...")
            bundles = _bundle_evidence_chunks(final_chunks, core_concepts, max_bundle_size=4)
            print(f"[BUNDLING] Created {len(bundles)} bundles")
            
            if bundles:
                for i, bundle in enumerate(bundles[:5]):
                    indicator_str = ", ".join(bundle["indicators"][:5]) if bundle["indicators"] else "none"
                    print(f"[BUNDLE {i+1}] Pages={bundle['pages']}, Score={bundle['bundle_score']:.3f}, Indicators=[{indicator_str}], Chunks={len(bundle['chunks'])}")
                    for j, chunk in enumerate(bundle["chunks"][:2]):
                        print(f"  [+] {chunk.get('id')} (def_score={chunk.get('definition_score', 0):.2f}, role={chunk.get('role', '?')})")
                
                # Use top 3-5 bundles and flatten to chunks
                top_bundles = bundles[:5]
                final_chunks = _flatten_bundles(top_bundles, max_total_chunks=10)
                print(f"[BUNDLING] Selected {len(final_chunks)} chunks from {len(top_bundles)} top bundles")

        # 6. Build prompt and generate answer
        prompt_faiss = _build_prompt(final_chunks, req.query)
        print(f"\n[ANSWER GENERATION] Prompt (first 300 chars): {prompt_faiss[:300]}")
        
        # 7. Generate LLM answer with confidence (INTENT-AWARE)
        llm_answer, confidence = _generate_llm_answer(req.query, final_chunks, question_intent)
        print(f"[ANSWER] {llm_answer[:200]}")
        print(f"[CONFIDENCE] {confidence:.2f}")
        
        # 8. Get ranked images for top text chunks
        top_text_ids = [t["id"] for t in final_chunks[:5]]
        ranked_images = GRAPH.get_ranked_images_for_texts(top_text_ids, limit=3)
        print(f"[IMAGES] Found {len(ranked_images)} related images")
        
        # 9. Store answer in Neo4j
        try:
            stored = GRAPH.store_answer(
                question=req.query,
                answer=llm_answer,
                confidence=confidence,
                text_ids=top_text_ids,
                image_ids=[img["id"] for img in ranked_images],
                metadata={
                    "intent": question_intent,
                    "core_concepts": core_concepts[:5],
                    "graph_validated": graph_validation["exists"],
                    "validated_pages": validated_pages[:10] if validated_pages else [],
                    "content_roles": list(set(content_roles)) if content_roles else [],
                    "evidence_matched": len(evidence_matched),
                    "evidence_rejected": len(evidence_rejected),
                }
            )
            print(f"[NEO4J] Answer stored: {stored}")
        except Exception as e:
            print(f"[NEO4J] Failed to store answer: {e}")
        
        # 10. NEW EVALUATION METRICS (replacing coverage/overlap)
        metrics = _metrics(llm_answer, final_chunks, core_concepts, question_intent)
        print(f"[METRICS] {metrics}")
        
        # 11. Return response
        return {
            "question": req.query,
            "answer": llm_answer,
            "abstained": False,
            "pass2_used": pass2_used,
            "confidence": confidence,
            "intent": question_intent,
            "core_concepts": core_concepts[:5],
            "metrics": metrics,  # NEW: Concept relevance, evidence-intent alignment, citation correctness
            "graph_validation": {
                "exists": graph_validation["exists"],
                "matched_concepts": graph_validation.get("matched_concepts", [])[:5],
                "validated_chunks": len(allowed_text_ids) if allowed_text_ids else 0,
                "pages": validated_pages[:10] if validated_pages else [],
                "roles": list(set(content_roles)) if content_roles else [],
                "match_quality": graph_validation.get("match_quality", "none"),
                "fallback_triggered": graph_validation.get("fallback_triggered", False),
            },
            "evidence_matching": {
                "total_retrieved": len(faiss_texts),
                "matched": len(evidence_matched),
                "rejected": len(evidence_rejected),
                "rejection_reasons": [r for _, r in evidence_rejected[:3]]
            },
            "pass2": {
                "attempted": pass2_used or (pass2_result is not None),
                "used": pass2_used,
                "expanded_queries": pass2_result.get("expanded_queries", [])[:5] if pass2_result else [],
                "restricted_pages": pass2_result.get("restricted_pages", [])[:10] if pass2_result else [],
                "candidates_found": pass2_result.get("candidate_count", 0) if pass2_result else 0,
            },
            "context_chunks": [
                {
                    "id": t["id"],
                    "page": t.get("page"),
                    "preview": t.get("content", "")[:200]
                }
                for t in final_chunks[:5]
            ],
            "images": [
                {
                    "id": img["id"],
                    "page": img.get("page"),
                    "url": f"/static/{Path(img['path']).name}" if img.get("path") else None
                }
                for img in ranked_images
            ]
        }
    
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/answers/recent")
def get_recent_answers(limit: int = 10):
    """
    Retrieve recent answers stored in Neo4j.
    Shows question, answer, confidence, and related text/image IDs.
    """
    try:
        answers = GRAPH.get_recent_answers(limit=limit)
        return {
            "count": len(answers),
            "answers": answers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.get("/image/{image_id}")
def get_image_by_id(image_id: str):
    """Serve local image file for a given Image node id."""
    try:
        rows = GRAPH.fetch_images([image_id])
        if not rows:
            raise HTTPException(status_code=404, detail="Image not found")
        p = Path(rows[0].get("path", "")).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {p}")
        return FileResponse(str(p))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
