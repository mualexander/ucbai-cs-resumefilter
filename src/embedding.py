"""Shared text-embedding backend for the full-text experiments (notebooks 07/08).

One definition of the embedder, its retry/backoff behavior, and the on-disk
embedding cache, so the two notebooks cannot drift apart (they previously held
diverging copies of this code). The cache filename includes backend, model,
and a content hash of the texts: changing any of them re-embeds; changing none
of them never does.

Spending is opt-in: with ``allow_api=False`` (the default) a cache miss raises
a ``RuntimeError`` instead of silently calling the API. ``force_reembed=True``
ignores existing cache files (and overwrites them) — only meaningful together
with ``allow_api=True``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.paths import EMBEDDINGS_DIR


def _l2norm(M):
    M = np.asarray(M, dtype=np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def make_token_counter():
    """tiktoken counter if available, else a whitespace-based length proxy."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s or ""))
    except Exception:
        return lambda s: int(len((s or "").split()) * 1.3)  # fallback length proxy


class Embedder:
    """Cache-aware sentence embedder with a swappable backend.

    backend="openai" (API; costs money) or "bge-m3" (local; costs time).
    Both backends L2-normalize, so cosine similarity is a plain dot product.
    Use the same instance for resumes AND job descriptions within a run;
    never mix backends across a comparison.
    """

    def __init__(self, backend: str = "openai",
                 openai_model: str = "text-embedding-3-small",
                 bge_model: str = "BAAI/bge-m3",
                 batch: int = 256,
                 request_token_budget: int = 200_000,   # stay under OpenAI's 300k tokens/request
                 max_item_tokens: int = 8000,           # text-embedding-3-small per-item limit is 8191
                 cache_dir=EMBEDDINGS_DIR,
                 allow_api: bool = False,
                 force_reembed: bool = False):
        self.backend = backend
        self.openai_model = openai_model
        self.bge_model = bge_model
        self.batch = batch
        self.request_token_budget = request_token_budget
        self.max_item_tokens = max_item_tokens
        self.cache_dir = cache_dir
        self.allow_api = allow_api
        self.force_reembed = force_reembed
        self._bge = None  # lazy SentenceTransformer singleton

    # ------------------------------------------------------------------ meta
    @property
    def model(self) -> str:
        return self.openai_model if self.backend == "openai" else self.bge_model

    @property
    def tag(self) -> str:
        """Cache-filename prefix; includes backend AND model so a model switch
        can never silently reuse another model's vectors."""
        return f"{self.backend}__{self.model}".replace("/", "_")

    # -------------------------------------------------------------- backends
    def _embed_openai(self, texts):
        import time, random
        from openai import OpenAI
        import openai as _oai
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
            def _ntok(s): return len(_enc.encode(s or ""))
            def _cut(s, mx):
                ids = _enc.encode(s or "")
                return _enc.decode(ids[:mx]) if len(ids) > mx else s
        except Exception:
            def _ntok(s): return int(len((s or "").split()) * 1.3)
            def _cut(s, mx):
                s = s or ""
                cap = int(mx * 2.5)   # conservative CHAR budget when no tokenizer is available
                return s[:cap] if len(s) > cap else s
        client = OpenAI(max_retries=0)   # we own the backoff so a 429 never aborts mid-corpus
        retryable = tuple(c for c in (getattr(_oai, "RateLimitError", None),
                                      getattr(_oai, "APITimeoutError", None),
                                      getattr(_oai, "APIConnectionError", None),
                                      getattr(_oai, "InternalServerError", None)) if c)

        def _wait(exc, fallback):
            h = getattr(getattr(exc, "response", None), "headers", None) or {}
            ms, s = h.get("retry-after-ms"), h.get("retry-after")
            try:
                if ms is not None:
                    return float(ms) / 1000.0
                if s is not None:
                    return float(s)
            except (TypeError, ValueError):
                pass
            return fallback

        def _is_token_limit(e):
            m = str(e).lower()
            return any(k in m for k in ("max_tokens_per_request", "maximum context length",
                                        "maximum input length", "reduce your", "8192"))

        # Self-correcting request: token estimates can undercount real text, so if the API rejects a
        # request for exceeding its per-request token cap, split the batch and retry recursively.
        def _embed_batch(batch):
            for attempt in range(8):
                try:
                    resp = client.embeddings.create(model=self.openai_model, input=batch)
                    return [d.embedding for d in resp.data]
                except retryable as e:
                    if attempt == 7:
                        raise
                    delay = _wait(e, 2.0 * (2 ** attempt)) + random.uniform(0.0, 0.5)
                    print(f"\n  rate-limited; backing off {delay:.1f}s")
                    time.sleep(delay)
                except Exception as e:
                    if _is_token_limit(e):
                        if len(batch) > 1:
                            mid = len(batch) // 2
                            print(f"\n  over token cap; splitting batch of {len(batch)} -> {mid}+{len(batch)-mid}")
                            return _embed_batch(batch[:mid]) + _embed_batch(batch[mid:])
                        s0 = batch[0]
                        batch = [s0[: max(256, len(s0) // 2)]]   # lone item too long -> halve by chars
                        continue
                    raise
            raise RuntimeError("embedding request failed after retries")

        # Per-item: truncate texts over the model's input limit (real resumes can be very long;
        # the synthetic corpora never hit this).
        prepped, n_trunc = [], 0
        for t in texts:
            t = t if (isinstance(t, str) and t.strip()) else " "
            tt = _cut(t, self.max_item_tokens)
            if tt != t:
                n_trunc += 1
            prepped.append((tt, _ntok(tt)))
        if n_trunc:
            print(f"  truncated {n_trunc} long text(s) to ~{self.max_item_tokens} tokens for embedding")

        # Per-request: pack to a (conservative) token budget; the split-on-error path is the safety net.
        out, i, n, done = [], 0, len(prepped), 0
        while i < n:
            batch, budget = [], 0
            while i < n and len(batch) < self.batch and (
                    not batch or budget + prepped[i][1] <= self.request_token_budget):
                batch.append(prepped[i][0])
                budget += prepped[i][1]
                i += 1
            out.extend(_embed_batch(batch))
            done += len(batch)
            print(f"  embedded {done}/{n}", end="\r")
        print()
        return _l2norm(out)

    def _embed_bge(self, texts):
        from sentence_transformers import SentenceTransformer
        if self._bge is None:
            self._bge = SentenceTransformer(self.bge_model)
        safe = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
        emb = self._bge.encode(safe, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float64)

    def embed_texts(self, texts):
        texts = list(texts)
        if self.backend == "openai":
            return self._embed_openai(texts)
        if self.backend == "bge-m3":
            return self._embed_bge(texts)
        raise ValueError(f"unknown backend {self.backend!r}")

    # ----------------------------------------------------------------- cache
    def corpus(self, texts, key: str):
        """Embed ``texts`` with an on-disk cache keyed by (backend, model, content hash)."""
        texts = list(texts)
        h = hashlib.sha256("\u0000".join(map(str, texts)).encode("utf-8")).hexdigest()[:12]
        cache = self.cache_dir / f"{self.tag}__{key}__{h}.npy"
        if cache.exists() and not self.force_reembed:
            M = np.load(cache)
            if M.shape[0] == len(texts):
                print(f"  [cache] {cache.name}")
                return M
        if not self.allow_api:
            raise RuntimeError(
                f"Embedding cache miss: {cache.name} (allow_api=False).\n"
                "Restore the cache file, or construct Embedder(allow_api=True) to (re)embed deliberately."
            )
        M = self.embed_texts(texts)
        np.save(cache, M)
        return M
