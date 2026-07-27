from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Ensure .env is loaded so Aura creds are available when module is imported
load_dotenv()

def _env(k, default=None):
    return os.getenv(k, default)

class GraphClient:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.uri = uri or _env("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or _env("NEO4J_USER", "neo4j")
        self.password = password or _env("NEO4J_PASSWORD", "neo4j")
        self.database = database or _env("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        print(f"[GRAPH CLIENT] Initialized with concept_expansion=enabled (BSL variants supported)")

    def close(self):
        self._driver.close()

    def neighbors(self, seed_id: str, limit: int = 50, rel_types: list = None):
        # Default: both SIMILAR_TO and HAS_IMAGE
        if rel_types is None:
            rel_types = ["SIMILAR_TO", "HAS_IMAGE"]
        
        # همسایه‌های دوطرفه، امتیاز اگر نبود صفر
        rel_filter = " OR ".join([f"type(r)='{rt}'" for rt in rel_types])
        q = f"""
        MATCH (n {{id: $id}})-[r]->(m)
        WHERE {rel_filter}
        RETURN m.id AS id,
            labels(m) AS labels,
            type(r) AS rel_type,
            coalesce(r.score, 0.0) AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as s:
            rows = s.run(q, id=seed_id, limit=limit).data()

        best = {}
        for r in rows:
            rid = r["id"]
            sc = float(r["score"] or 0.0)
            if rid not in best or best[rid]["score"] < sc:
                best[rid] = {
                    "id": rid,
                    "labels": r["labels"],
                    "rel_type": r["rel_type"],
                    "score": sc
                }
        return list(best.values())

    def fetch_text_chunks(self, ids):
        if not ids:
            return []
        q = """
        MATCH (t:Text) WHERE t.id IN $ids
        RETURN t.id AS id, t.page AS page, t.text AS text
        """
        with self._driver.session(database=self.database) as s:
            return s.run(q, ids=list(ids)).data()

    def fetch_images(self, ids):
        if not ids:
            return []
        q = """
        MATCH (i:Image) WHERE i.id IN $ids
        RETURN i.id AS id, i.page AS page, i.path AS path
        """
        with self._driver.session(database=self.database) as s:
            return s.run(q, ids=list(ids)).data()
    
    def get_ranked_images_for_texts(self, text_ids, limit=5):
        """Get images related to given text chunks, ranked by relationship score"""
        if not text_ids:
            return []
        q = """
        MATCH (t:Text)-[r:HAS_IMAGE]->(i:Image)
        WHERE t.id IN $text_ids
        RETURN DISTINCT i.id AS id, i.page AS page, i.path AS path,
               MAX(coalesce(r.score, 0.0)) AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as s:
            return s.run(q, text_ids=list(text_ids), limit=limit).data()
    
    def store_answer(self, question, answer, confidence, text_ids, image_ids, metadata=None):
        """Store generated answer with references to text chunks and images"""
        import datetime
        import json
        q = """
        CREATE (a:Answer {
            question: $question,
            answer: $answer,
            confidence: $confidence,
            timestamp: datetime($timestamp),
            metadata: $metadata
        })
        WITH a
        UNWIND $text_ids AS text_id
        MATCH (t:Text {id: text_id})
        CREATE (a)-[:BASED_ON_TEXT]->(t)
        WITH a
        UNWIND $image_ids AS image_id
        MATCH (i:Image {id: image_id})
        CREATE (a)-[:INCLUDES_IMAGE]->(i)
        RETURN a.question AS question, a.answer AS answer
        """
        with self._driver.session(database=self.database) as s:
            result = s.run(
                q,
                question=question,
                answer=answer,
                confidence=float(confidence),
                timestamp=datetime.datetime.now().isoformat(),
                metadata=json.dumps(metadata) if metadata else "{}",
                text_ids=list(text_ids) if text_ids else [],
                image_ids=list(image_ids) if image_ids else []
            ).data()
            return result[0] if result else None
    
    def get_recent_answers(self, limit=10):
        """Retrieve recent stored answers with their references"""
        q = """
        MATCH (a:Answer)
        OPTIONAL MATCH (a)-[:BASED_ON_TEXT]->(t:Text)
        OPTIONAL MATCH (a)-[:INCLUDES_IMAGE]->(i:Image)
        WITH a, collect(DISTINCT t.id) AS text_ids, collect(DISTINCT i.id) AS image_ids
        RETURN a.question AS question,
               a.answer AS answer,
               a.confidence AS confidence,
               a.timestamp AS timestamp,
               text_ids,
               image_ids
        ORDER BY a.timestamp DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as s:
            return s.run(q, limit=limit).data()    
    def validate_concepts(self, concepts, entities):
        """
        PHRASE/ALIAS-BASED concept validation with soft fallbacks.
        
        Strategy:
        1. Generate canonical + aliases (phrase-aware, not token-based)
        2. Query Neo4j for LARGE candidate set (50-200 chunks)
        3. Rank by match type: phrase > alias > section-level fallback
        4. Fallback chain: phrase match → alias match → biosafety pages + FAISS
        
        Returns: {
            "exists": bool,
            "matched_concepts": [str],  # canonical concepts that matched
            "text_ids": [str],          # ranked chunk IDs
            "pages": [int],
            "roles": [str],
            "match_quality": str,       # "phrase" | "alias" | "section" | "none"
            "fallback_triggered": bool,
        }
        """
        import re
        
        if not concepts and not entities:
            return {
                "exists": False,
                "matched_concepts": [],
                "text_ids": [],
                "pages": [],
                "roles": [],
                "match_quality": "none",
                "fallback_triggered": False,
            }
        
        # Stopwords to ignore in single-word matching
        STOPWORDS = {"level", "a", "the", "is", "are", "by", "of", "in", "on", "to", "for"}
        
        def _generate_phrase_and_aliases(term: str):
            """
            Generate canonical phrase and its aliases (NOT individual tokens).
            Returns: {"canonical": str, "aliases": [str]}
            """
            term = term.lower().strip()
            canonical = term
            aliases = []
            
            # Skip pure stopwords
            if term in STOPWORDS:
                return {"canonical": None, "aliases": []}
            
            # BSL-level expansions (phrase-based, not token)
            if any(w in term for w in ["biosafety level", "bsl", "containment level"]):
                # Match number at END of term, after space or hyphen (not in middle of word)
                num_match = re.search(r"[\s-](\d+|ii|iii|iv|i)$", term)
                num = num_match.group(1) if num_match else None
                
                # Also try matching if term IS just the level (e.g., "level 2", "bsl-2")
                if not num and re.match(r"(level|bsl|containment level)[\s-]?(\d+|ii|iii|iv|i)$", term):
                    num = re.search(r"(\d+|ii|iii|iv|i)$", term).group(1)
                
                if num:
                    # Generate canonical: "biosafety level N"
                    canonical = f"biosafety level {num}"
                    aliases.extend([
                        f"biosafety level-{num}",
                        f"bsl-{num}",
                        f"bsl {num}",
                        f"bsl{num}",
                        f"containment level {num}",
                        f"containment level-{num}",
                        f"level {num}",  # Allow "level N" if N is numeric
                    ])
                else:
                    canonical = "biosafety level"
                    aliases.extend(["bsl", "containment level"])
            
            # Remove duplicates
            aliases = [a for a in aliases if a and a != canonical]
            return {"canonical": canonical, "aliases": list(set(aliases))}
        
        # Build search space: canonical + aliases
        print(f"[CONCEPT PROCESSING] Raw input: concepts={concepts}, entities={entities}")
        
        all_inputs = list(concepts) + list(entities)
        search_space = {}  # canonical -> aliases
        for term in all_inputs:
            expanded = _generate_phrase_and_aliases(term)
            if expanded["canonical"]:
                search_space[expanded["canonical"]] = expanded["aliases"]
        
        print(f"[CONCEPT PROCESSING] After normalization: {search_space}")
        
        # Build regex patterns for phrase matching (prioritize multi-word)
        phrase_patterns = list(search_space.keys())
        phrase_patterns.sort(key=lambda x: -len(x.split()))  # Multi-word first
        
        alias_patterns = []
        for aliases in search_space.values():
            alias_patterns.extend(aliases)
        alias_patterns = list(set(alias_patterns))
        alias_patterns.sort(key=lambda x: -len(x.split()))  # Multi-word first
        
        print(f"[CONCEPT PROCESSING] Phrase patterns (top 5): {phrase_patterns[:5]}")
        print(f"[CONCEPT PROCESSING] Alias patterns (top 5): {alias_patterns[:5]}")
        
        # Query Neo4j: retrieve large candidate set (200 chunks)
        def _query_by_patterns(patterns):
            """Query Neo4j for chunks matching patterns (hyphen/space agnostic)."""
            if not patterns:
                return []
            
            regex_patterns = []
            for p in patterns[:30]:  # Limit to 30 to avoid query explosion
                # Make regex hyphen/space agnostic for multi-word phrases
                escaped = re.escape(p)
                escaped_agnostic = escaped.replace("\\ ", "[\\s-]+")
                regex_patterns.append(f"(?i).*{escaped_agnostic}.*")
            
            q = """
            MATCH (t:Text)
            WHERE ANY(pattern IN $patterns WHERE t.text =~ pattern)
            RETURN t.id AS id, 
                   t.page AS page, 
                   t.text AS text
            LIMIT $limit
            """
            
            with self._driver.session(database=self.database) as s:
                return s.run(q, patterns=regex_patterns, limit=200).data()
        
        # Attempt 1: Match by PHRASE
        print("[CONCEPT PROCESSING] Attempt 1: Phrase matching...")
        phrase_results = _query_by_patterns(phrase_patterns)
        
        if phrase_results:
            print(f"[CONCEPT PROCESSING] [+] Phrase match found: {len(phrase_results)} chunks")
            matched_text_ids = []
            matched_pages = set()
            matched_roles = []
            matched_phrases = set()
            
            for r in phrase_results:
                text_id = r["id"]
                page = r["page"]
                text_content = r["text"]
                
                # Verify phrase match in actual content
                has_phrase = False
                for phrase in phrase_patterns:
                    esc = re.escape(phrase).replace("\\ ", "[\\s-]+")
                    if re.search(rf"(?i)\b{esc}\b", text_content.lower()):
                        matched_phrases.add(phrase)
                        has_phrase = True
                        break
                
                if has_phrase:
                    matched_text_ids.append(text_id)
                    matched_pages.add(page)
                    role = self._classify_content_role(text_content)
                    matched_roles.append(role)
            
            if matched_text_ids:
                return {
                    "exists": True,
                    "matched_concepts": list(matched_phrases),
                    "text_ids": matched_text_ids,
                    "pages": sorted(list(matched_pages)),
                    "roles": matched_roles,
                    "match_quality": "phrase",
                    "fallback_triggered": False,
                }
        
        # Attempt 2: Match by ALIAS
        print("[CONCEPT PROCESSING] Attempt 2: Alias matching...")
        alias_results = _query_by_patterns(alias_patterns)
        
        if alias_results:
            print(f"[CONCEPT PROCESSING] [+] Alias match found: {len(alias_results)} chunks")
            matched_text_ids = []
            matched_pages = set()
            matched_roles = []
            matched_aliases = set()
            
            for r in alias_results:
                text_id = r["id"]
                page = r["page"]
                text_content = r["text"]
                
                # Verify alias match
                has_alias = False
                for alias in alias_patterns:
                    esc = re.escape(alias).replace("\\ ", "[\\s-]+")
                    if re.search(rf"(?i)\b{esc}\b", text_content.lower()):
                        matched_aliases.add(alias)
                        has_alias = True
                        break
                
                if has_alias:
                    matched_text_ids.append(text_id)
                    matched_pages.add(page)
                    role = self._classify_content_role(text_content)
                    matched_roles.append(role)
            
            if matched_text_ids:
                return {
                    "exists": True,
                    "matched_concepts": list(matched_aliases),
                    "text_ids": matched_text_ids,
                    "pages": sorted(list(matched_pages)),
                    "roles": matched_roles,
                    "match_quality": "alias",
                    "fallback_triggered": True,
                }
        
        # Attempt 3: Fallback to SECTION-LEVEL (biosafety pages)
        print("[CONCEPT PROCESSING] Attempt 3: Section-level fallback to biosafety pages...")
        q_fallback = """
        MATCH (t:Text)
        WHERE t.text =~ '(?i).*biosafety.*'
        RETURN t.id AS id, t.page AS page, t.text AS text
        LIMIT 200
        """
        
        with self._driver.session(database=self.database) as s:
            fallback_results = s.run(q_fallback).data()
        
        if fallback_results:
            print(f"[CONCEPT PROCESSING] [+] Section fallback: {len(fallback_results)} biosafety pages")
            matched_text_ids = [r["id"] for r in fallback_results]
            matched_pages = sorted(set(r["page"] for r in fallback_results))
            matched_roles = [self._classify_content_role(r["text"]) for r in fallback_results]
            
            return {
                "exists": True,
                "matched_concepts": ["biosafety"],  # Generic concept
                "text_ids": matched_text_ids,
                "pages": matched_pages,
                "roles": matched_roles,
                "match_quality": "section",
                "fallback_triggered": True,
            }
        
        # No matches at all
        print("[CONCEPT PROCESSING] ✗ No matches found at any level")
        return {
            "exists": False,
            "matched_concepts": [],
            "text_ids": [],
            "pages": [],
            "roles": [],
            "match_quality": "none",
            "fallback_triggered": False,
        }
    
    def _classify_content_role(self, text: str):
        """Classify text chunk role: definition, procedure, table, example"""
        text_lower = text.lower()
        
        # Definition patterns (strict)
        if any(p in text_lower for p in [" is defined as", " refers to ", " is a ", " are ", " means ", " definition ", " is the ", " are the ", " known as "]):
            return "definition"
        # Also allow explicit BSL-style definitional phrasing
        import re
        if re.search(r"\blevel\s+\d+\s+laborator", text_lower):
            return "definition"
        
        # Procedure patterns
        if any(p in text_lower for p in ["step ", "procedure", "method", "how to", "should be", "must be", "process"]):
            return "procedure"
        
        # Table patterns (has lots of numbers, dashes, structured data)
        if text.count('\t') > 3 or text.count('|') > 3 or (text.count('-') > 10 and text.count('\n') > 5):
            return "table"
        
        # Example patterns
        if any(p in text_lower for p in ["for example", "e.g.", "such as", "example:", "instance"]):
            return "example"
        
        return "general"