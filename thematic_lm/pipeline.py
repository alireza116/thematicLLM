"""
ThematicLMPipeline — orchestrates the full two-stage pipeline from the paper:

  Stage 1 (Coding):
    For each batch of data items:
      1. N coder agents independently produce codes + quotes.
      2. Code aggregator merges coder outputs.
      3. Reviewer retrieves top-k similar codes from the codebook, decides
         how to update/merge entries, then writes to the codebook.

  Stage 2 (Theme Development):
    1. M theme coder agents each receive the full (compressed) codebook and
       independently identify themes.
    2. Theme aggregator merges them into the final theme set.
"""

from __future__ import annotations
import json
import os
from typing import Optional
from tqdm import tqdm

from .llm import LLMClient
from .codebook import Codebook
from .agents import (
    coder_agent,
    coder_agent_batch,
    code_aggregator,
    reviewer_agent,
    theme_coder_agent,
    theme_aggregator,
)


class ThematicLMPipeline:
    def __init__(
        self,
        client: LLMClient,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_coders: int = 2,
        n_theme_coders: int = 2,
        coder_identities: Optional[list[Optional[str]]] = None,
        theme_coder_identities: Optional[list[Optional[str]]] = None,
        top_k_quotes: int = 20,
        top_k_similar: int = 10,
        batch_size: int = 10,
        coder_batch_size: int = 20,
        codebook_path: Optional[str] = None,
        study_context: Optional[str] = None,
    ):
        """
        Args:
            client: LLMClient (OpenAI, Anthropic, or compatible).
            embedding_model_name: SentenceTransformer model for code embeddings.
            n_coders: Number of parallel coder agents in Stage 1.
            n_theme_coders: Number of parallel theme coder agents in Stage 2.
            coder_identities: List of identity strings for coders (len == n_coders).
                              Use None entries for no-identity coders.
            theme_coder_identities: Same for theme coders.
            top_k_quotes: Max quotes stored per code/theme.
            top_k_similar: Number of similar codes the reviewer retrieves.
            batch_size: Items processed per aggregation batch (goes to reviewer).
            coder_batch_size: Items sent to coder in a single API call.
            codebook_path: If set, save/load the codebook from this JSON file.
            study_context: Optional description of the study and survey questions,
                           injected into coder and theme coder system prompts.
        """
        self.client = client
        self.study_context = study_context
        self.n_coders = n_coders
        self.n_theme_coders = n_theme_coders
        self.top_k_quotes = top_k_quotes
        self.top_k_similar = top_k_similar
        self.batch_size = batch_size
        self.coder_batch_size = coder_batch_size
        self.codebook_path = codebook_path

        # Pad / truncate identity lists to match coder counts
        self.coder_identities = self._align_identities(coder_identities, n_coders)
        self.theme_coder_identities = self._align_identities(theme_coder_identities, n_theme_coders)

        # Load embedding model
        print(f"Loading embedding model '{embedding_model_name}'...")
        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer(embedding_model_name)

        # Init (or load) codebook
        if codebook_path and os.path.exists(codebook_path):
            print(f"Resuming codebook from {codebook_path}")
            self.codebook = Codebook.load(codebook_path, self._embedding_model)
        else:
            self.codebook = Codebook(embedding_model=self._embedding_model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_identities(identities, n):
        if identities is None:
            return [None] * n
        if len(identities) >= n:
            return list(identities[:n])
        # Cycle through if fewer identities than coders
        return [identities[i % len(identities)] for i in range(n)]

    def _apply_reviewer_decisions(
        self,
        decisions: list[dict],
        aggregated_codes: list[dict],
    ) -> None:
        """Apply reviewer decisions to the codebook."""
        # Build a lookup for aggregated codes by label
        agg_lookup = {c["code"]: c for c in aggregated_codes}

        for decision in decisions:
            new_code = decision.get("new_code", "")
            action = decision.get("action", "keep")
            final_code = decision.get("final_code", new_code)
            merge_with = decision.get("merge_with", [])

            source = agg_lookup.get(new_code, {})
            quotes = source.get("quotes", [])
            quote_ids = source.get("quote_ids", [])

            if action == "keep":
                for q, qid in zip(quotes, quote_ids):
                    self.codebook.add_code(final_code, q, qid)

            elif action == "update":
                # Store under the refined name
                for q, qid in zip(quotes, quote_ids):
                    self.codebook.add_code(final_code, q, qid)

            elif action == "merge":
                # Add all quotes to the merged code, then merge in codebook
                for q, qid in zip(quotes, quote_ids):
                    self.codebook.add_code(final_code, q, qid)
                if merge_with:
                    all_codes = [new_code] + merge_with
                    self.codebook.merge_codes(all_codes, final_code)

        self.codebook.trim_quotes(self.top_k_quotes)

    # ------------------------------------------------------------------
    # Stage 1: Coding
    # ------------------------------------------------------------------

    def run_coding_stage(self, data: list[dict]) -> Codebook:
        """
        Process all data items through the coding pipeline.

        Args:
            data: List of {"id": str, "text": str} dicts.

        Returns:
            The updated Codebook.
        """
        print(f"\n=== Stage 1: Coding ({len(data)} items, batch_size={self.batch_size}) ===")

        for batch_start in tqdm(range(0, len(data), self.batch_size), desc="Batches"):
            batch = data[batch_start : batch_start + self.batch_size]

            # Collect coder outputs across the batch using mini-batch calls
            all_coder_outputs: list[list[dict]] = [[] for _ in range(self.n_coders)]

            # Normalize items so each has a string id
            batch_items = [
                {"id": str(item.get("id", batch_start + i)), "text": item.get("text", "")}
                for i, item in enumerate(batch)
            ]

            for coder_idx in range(self.n_coders):
                identity = self.coder_identities[coder_idx]
                # Send items to the coder in mini-batches
                for mini_start in range(0, len(batch_items), self.coder_batch_size):
                    mini_batch = batch_items[mini_start : mini_start + self.coder_batch_size]
                    codes = coder_agent_batch(self.client, mini_batch, identity, self.study_context)
                    all_coder_outputs[coder_idx].extend(codes)

            # Aggregate codes from all coders in this batch
            aggregated = code_aggregator(self.client, all_coder_outputs, self.top_k_quotes)

            if not aggregated:
                continue

            # Build similar-code lookup for the reviewer
            similar_codes_payload = []
            for agg_code in aggregated:
                similar = self.codebook.get_similar_codes(
                    agg_code["code"], top_k=self.top_k_similar
                )
                similar_codes_payload.append({
                    "new_code": agg_code["code"],
                    "similar": similar,
                })

            # Reviewer decides how to update codebook
            decisions = reviewer_agent(self.client, aggregated, similar_codes_payload)

            # Apply decisions
            self._apply_reviewer_decisions(decisions, aggregated)

            # Periodically save codebook
            if self.codebook_path:
                self.codebook.save(self.codebook_path)

        if self.codebook_path:
            self.codebook.save(self.codebook_path)

        print(f"Codebook finalized: {len(self.codebook)} codes")
        return self.codebook

    # ------------------------------------------------------------------
    # Stage 2: Theme Development
    # ------------------------------------------------------------------

    def run_theme_stage(self) -> list[dict]:
        """
        Run theme development on the current codebook.

        Returns:
            Final list of theme dicts {"theme", "description", "quotes", "quote_ids"}.
        """
        print(f"\n=== Stage 2: Theme Development ({self.n_theme_coders} theme coders) ===")

        codebook_json = self.codebook.to_json()

        all_theme_outputs = []
        for i in range(self.n_theme_coders):
            identity = self.theme_coder_identities[i]
            print(f"Theme coder {i + 1}/{self.n_theme_coders}...")
            themes = theme_coder_agent(
                self.client,
                codebook_json,
                identity=identity,
                top_k=self.top_k_quotes,
                study_context=self.study_context,
            )
            all_theme_outputs.append(themes)

        print("Aggregating themes...")
        final_themes = theme_aggregator(self.client, all_theme_outputs, self.top_k_quotes)
        return final_themes

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, data: list[dict]) -> list[dict]:
        """
        Run the complete Thematic-LM pipeline.

        Args:
            data: List of {"id": str, "text": str} dicts.

        Returns:
            Final themes as a list of dicts.
        """
        self.run_coding_stage(data)
        themes = self.run_theme_stage()
        return themes

    # ------------------------------------------------------------------
    # Convenience: run on pre-split data for transferability evaluation
    # ------------------------------------------------------------------

    def run_split(
        self,
        data: list[dict],
        split: float = 0.5,
    ) -> tuple[list[dict], list[dict]]:
        """
        Split data and run the pipeline on each half independently.
        Returns (train_themes, val_themes) for transferability evaluation.
        """
        split_idx = int(len(data) * split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        print("Running on training split...")
        train_pipeline = ThematicLMPipeline(
            client=self.client,
            n_coders=self.n_coders,
            n_theme_coders=self.n_theme_coders,
            coder_identities=self.coder_identities,
            theme_coder_identities=self.theme_coder_identities,
            top_k_quotes=self.top_k_quotes,
            top_k_similar=self.top_k_similar,
            batch_size=self.batch_size,
        )
        train_pipeline._embedding_model = self._embedding_model
        train_pipeline.codebook = Codebook(embedding_model=self._embedding_model)
        train_themes = train_pipeline.run(train_data)

        print("Running on validation split...")
        val_pipeline = ThematicLMPipeline(
            client=self.client,
            n_coders=self.n_coders,
            n_theme_coders=self.n_theme_coders,
            coder_identities=self.coder_identities,
            theme_coder_identities=self.theme_coder_identities,
            top_k_quotes=self.top_k_quotes,
            top_k_similar=self.top_k_similar,
            batch_size=self.batch_size,
        )
        val_pipeline._embedding_model = self._embedding_model
        val_pipeline.codebook = Codebook(embedding_model=self._embedding_model)
        val_themes = val_pipeline.run(val_data)

        return train_themes, val_themes
