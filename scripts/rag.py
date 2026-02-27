# rag.py - Simple RAG robusto y tolerante con diferentes formatos de 'docs'
from typing import List, Tuple, Dict, Any
import numpy as np
import re

# Intento de usar sklearn; si no está, usar fallback simple
_USE_SKLEARN = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    _USE_SKLEARN = True
except Exception as e:
    print("rag.py: sklearn no disponible - usando fallback simple. Error:", e)
    _USE_SKLEARN = False

def _simple_tokenize(s: str):
    if not s:
        return []
    s2 = re.sub(r"[^\w\s]", " ", s.lower(), flags=re.UNICODE)
    tokens = [t for t in s2.split() if len(t) > 1]
    return tokens

def _jaccard_score(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))

class SimpleRAG:
    """
    Simple RAG tolerante:
     - Normaliza distintos formatos de 'docs' a una lista de (id,text)
     - Usa TF-IDF si está disponible, si no usa Jaccard/token overlap como fallback
     - build_context(q, top_k, token_budget_chars) -> (context_str, sources)
       donde sources = [ {"id":..., "score":..., "meta": {...}}, ... ]
    """

    def __init__(self, docs: Any):
        """
        docs puede ser:
         - list of (id, text)
         - list of dicts [{'id':..., 'text':...}, {'source':...,'content':...}, ...]
         - dict {id: text, ...}
         - list of strings [text1, text2,...] (generará ids automáticos)
        """
        # Normalizar entrada a lista de tuplas (id, text)
        self.docs_raw = docs
        self.normalized: List[Tuple[str,str]] = self._normalize_docs(docs)
        self.ids = [d[0] for d in self.normalized]
        self.texts = [d[1] or "" for d in self.normalized]

        # preparar sklearn si disponible
        self.use_sklearn = False
        self.matrix = None
        if _USE_SKLEARN and len(self.texts) > 0:
            try:
                # Intentar vectorizer con stop_words en inglés (si prefieres español, cambia)
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.matrix = self.vectorizer.fit_transform(self.texts)
                self.use_sklearn = True
                print(f"rag.py: TF-IDF inicializado con {len(self.texts)} documentos (sklearn).")
            except Exception as e:
                print("rag.py: fallo al inicializar TF-IDF, cayendo a fallback. Error:", e)
                self.use_sklearn = False
                self.matrix = None
        else:
            if len(self.texts) == 0:
                print("rag.py: no se han proporcionado textos (docs vacío).")
            else:
                print("rag.py: sklearn no disponible; usando fallback token-jaccard.")
            self.use_sklearn = False

        # pre-tokenizar para fallback
        self._token_texts = [_simple_tokenize(t) for t in self.texts]

    def _normalize_docs(self, docs_input: Any) -> List[Tuple[str,str]]:
        """
        Intenta convertir 'docs_input' en lista de (id, text).
        """
        out: List[Tuple[str,str]] = []
        if docs_input is None:
            return out

        # si es un dict {id: text}
        if isinstance(docs_input, dict):
            for k, v in docs_input.items():
                out.append((str(k), str(v or "")))
            return out

        # si es lista/iterable
        try:
            # comprobar si es lista de tuplas (id,text)
            if isinstance(docs_input, (list, tuple)):
                for idx, item in enumerate(docs_input):
                    # caso: (id, text)
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        id0 = str(item[0])
                        text0 = str(item[1] or "")
                        out.append((id0, text0))
                        continue
                    # caso: dict con keys id/text o source/content
                    if isinstance(item, dict):
                        # buscar claves comunes
                        idk = None
                        textk = None
                        for k in ("id", "source", "filename", "name"):
                            if k in item:
                                idk = str(item[k])
                                break
                        for k in ("text", "content", "body", "snippet"):
                            if k in item:
                                textk = str(item[k] or "")
                                break
                        # si no hay id, usar índice como id
                        if idk is None:
                            idk = f"doc_{idx}"
                        if textk is None:
                            # si dict tiene sólo un valor que no es id, intentar cogerlo
                            # por ejemplo {'doc1':'texto...'}
                            vals = [v for k,v in item.items() if k not in ("id","source","text","content","body","snippet","filename","name")]
                            if vals:
                                textk = str(vals[0] or "")
                            else:
                                textk = ""
                        out.append((idk, textk))
                        continue
                    # caso: string (texto sin id)
                    if isinstance(item, str):
                        out.append((f"doc_{idx}", item))
                        continue
                    # caso: objeto con __dict__
                    if hasattr(item, "__dict__"):
                        d = getattr(item, "__dict__")
                        idk = d.get("id") or d.get("source") or f"doc_{idx}"
                        textk = d.get("text") or d.get("content") or ""
                        out.append((str(idk), str(textk)))
                        continue
                    # fallback: convertir a str
                    out.append((f"doc_{idx}", str(item)))
                return out
        except Exception as e:
            print("rag.py: excepción al normalizar docs_input:", e)
            # intentar convertir iterando y forzando str
            try:
                for idx, item in enumerate(docs_input):
                    out.append((f"doc_{idx}", str(item)))
                return out
            except Exception:
                return out

        # si no coincide ningún tipo, intentar str(docs_input)
        try:
            out.append(("doc_0", str(docs_input)))
        except Exception:
            pass
        return out

    def _parse_source_meta(self, source_id: str) -> Dict[str,Any]:
        meta = {"source": source_id, "chunk": None}
        try:
            if "::" in source_id:
                fname, rest = source_id.split("::", 1)
                meta["source"] = fname
                if "chunk_" in rest:
                    try:
                        num = rest.split("chunk_")[-1]
                        meta["chunk"] = int(num)
                    except Exception:
                        meta["chunk"] = rest
                else:
                    meta["chunk"] = rest
        except Exception:
            pass
        return meta

    def query(self, q: str, top_k: int = 4) -> List[Tuple[str,float,str]]:
        q = q or ""
        top_k = max(1, int(top_k or 1))
        results: List[Tuple[str,float,str]] = []

        if len(self.texts) == 0:
            return results

        # Try sklearn TF-IDF approach
        if self.use_sklearn and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([q])
                sim = linear_kernel(qv, self.matrix).flatten()
                idxs = np.argsort(-sim)[:top_k]
                for i in idxs:
                    score = float(sim[i])
                    if score > 0:
                        results.append((self.ids[i], score, self.texts[i]))
                return results
            except Exception as e:
                print("rag.py: error en query sklearn, usando fallback. Error:", e)

        # Fallback: Jaccard/token overlap
        q_tokens = _simple_tokenize(q)
        scores = []
        for i, tokens in enumerate(self._token_texts):
            score = _jaccard_score(q_tokens, tokens)
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append((self.ids[idx], float(score), self.texts[idx]))
        return results

    def build_context(self, q: str, top_k: int = 6, token_budget_chars: int = 1200) -> Tuple[str, List[Dict[str,Any]]]:
        context_fragments: List[str] = []
        sources: List[Dict[str,Any]] = []

        try:
            hits = self.query(q, top_k=top_k)
        except Exception as e:
            print("rag.py: error en query, devolviendo vacío. Error:", e)
            hits = []

        if not hits:
            return ("", [])

        total_chars = 0
        for sid, score, txt in hits:
            txt = txt or ""
            if token_budget_chars and (total_chars + len(txt) > token_budget_chars):
                if total_chars == 0:
                    snippet = txt[:max(200, token_budget_chars)]
                    context_fragments.append(f"FUENTE: {sid}\n{snippet}\n---\n")
                    total_chars += len(snippet)
                break
            context_fragments.append(f"FUENTE: {sid}\n{txt}\n---\n")
            total_chars += len(txt)
            meta = self._parse_source_meta(sid)
            sources.append({"id": sid, "score": float(score), "meta": meta})

        context_str = "\n".join(context_fragments).strip()
        return (context_str, sources)