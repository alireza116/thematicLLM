"""
Adaptive codebook that stores codes, quotes, and embeddings.
The reviewer agent uses this to retrieve similar codes and update entries.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


class Codebook:
    def __init__(self, embedding_model=None):
        # {code_text: {"quotes": [...], "quote_ids": [...], "embedding": np.array}}
        self._codes: dict[str, dict] = {}
        self._model = embedding_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("No embedding model provided to Codebook.")
        return self._model.encode([text])[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_code(self, code: str, quote: str, quote_id: str) -> None:
        """Add a new code or append a quote to an existing one."""
        if code not in self._codes:
            self._codes[code] = {
                "quotes": [],
                "quote_ids": [],
                "embedding": self._embed(code).tolist(),
            }
        entry = self._codes[code]
        if quote_id not in entry["quote_ids"]:
            entry["quotes"].append(quote)
            entry["quote_ids"].append(quote_id)

    def update_code(self, old_code: str, new_code: str) -> None:
        """Rename a code (e.g. after reviewer refinement)."""
        if old_code not in self._codes:
            return
        entry = self._codes.pop(old_code)
        entry["embedding"] = self._embed(new_code).tolist()
        if new_code in self._codes:
            # Merge into existing entry
            self._codes[new_code]["quotes"].extend(entry["quotes"])
            self._codes[new_code]["quote_ids"].extend(entry["quote_ids"])
        else:
            self._codes[new_code] = entry

    def merge_codes(self, codes_to_merge: list[str], merged_name: str) -> None:
        """Merge multiple codes into one."""
        combined_quotes, combined_ids = [], []
        for code in codes_to_merge:
            if code in self._codes:
                entry = self._codes.pop(code)
                combined_quotes.extend(entry["quotes"])
                combined_ids.extend(entry["quote_ids"])
        if merged_name in self._codes:
            self._codes[merged_name]["quotes"].extend(combined_quotes)
            self._codes[merged_name]["quote_ids"].extend(combined_ids)
        else:
            self._codes[merged_name] = {
                "quotes": combined_quotes,
                "quote_ids": combined_ids,
                "embedding": self._embed(merged_name).tolist(),
            }

    def get_similar_codes(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Return top-k most similar codes by cosine similarity."""
        if not self._codes:
            return []
        query_emb = self._embed(query_text).reshape(1, -1)
        codes = list(self._codes.keys())
        embeddings = np.array([self._codes[c]["embedding"] for c in codes])
        sims = cosine_similarity(query_emb, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "code": codes[i],
                "similarity": float(sims[i]),
                "quotes": self._codes[codes[i]]["quotes"][:5],
                "quote_ids": self._codes[codes[i]]["quote_ids"][:5],
            }
            for i in top_indices
        ]

    def trim_quotes(self, max_quotes: int = 20) -> None:
        """Keep only the most recent N quotes per code to control size."""
        for entry in self._codes.values():
            entry["quotes"] = entry["quotes"][-max_quotes:]
            entry["quote_ids"] = entry["quote_ids"][-max_quotes:]

    def to_dict(self) -> dict:
        return {
            code: {
                "quotes": data["quotes"],
                "quote_ids": data["quote_ids"],
            }
            for code, data in self._codes.items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def codes(self) -> list[str]:
        return list(self._codes.keys())

    def __len__(self) -> int:
        return len(self._codes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            code: {
                "quotes": entry["quotes"],
                "quote_ids": entry["quote_ids"],
                "embedding": entry["embedding"],
            }
            for code, entry in self._codes.items()
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str, embedding_model=None) -> "Codebook":
        cb = cls(embedding_model=embedding_model)
        data = json.loads(Path(path).read_text())
        for code, entry in data.items():
            cb._codes[code] = {
                "quotes": entry["quotes"],
                "quote_ids": entry["quote_ids"],
                "embedding": entry["embedding"],
            }
        return cb
