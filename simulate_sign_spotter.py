#!/usr/bin/env python3
"""
simulate_sign_spotter.py

Simulate a corpus of sign gloss sequences and an imperfect sign-spotter that
produces spotted tokens and an estimated document length (automatic segmenter).

Outputs a corrupted JSONL corpus with fields:
  - doc_id: str
  - true_tokens: List[str]
  - spotted_tokens: List[str]
  - estimated_length: int

Also provides utilities to compute DF / avgdl and a simple BM25 scorer
to compare scoring on true vs spotted token corpora.

Author: ChatGPT (meticulous-python style)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from tqdm import tqdm

# -------------------------
# Logging / defaults
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("simulate_sign_spotter")


# -------------------------
# Data classes
# -------------------------
@dataclass(frozen=True)
class SpotterConfig:
    """Configuration for the simulated sign-spotter."""

    spotter_vocab: Sequence[str]
    p_false_negative: float = 0.1
    p_false_positive: float = 0.02
    p_substitution: float = 0.0
    per_gloss_fn: Optional[Mapping[str, float]] = None
    per_gloss_fp: Optional[Mapping[str, float]] = None
    seed: int = 42


class LengthEstimatorConfig:
    def __init__(
        self,
        sigma: float = 0.0,
        bias: float = 0.0,
        min_len: int = 1,
        max_len: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.sigma = sigma
        self.bias = bias
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed


# -------------------------
# Utilities
# -------------------------
def safe_choice(seq: Sequence[str], rnd: random.Random) -> str:
    """Return a random element; if sequence empty raise ValueError."""
    if not seq:
        raise ValueError("safe_choice: sequence is empty")
    return rnd.choice(seq)


def compute_df_and_avgdl(docs: Iterable[Sequence[str]]) -> Tuple[Dict[str, int], float]:
    """Compute document frequency (DF) per token and average document length (avgdl)."""
    df: MutableMapping[str, int] = {}
    lens: List[int] = []
    for tokens in docs:
        lens.append(len(tokens))
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    avgdl = mean(lens) if lens else 0.0
    return dict(df), avgdl


# -------------------------
# Spotter simulation
# -------------------------
def simulate_spotter_on_doc(
    true_tokens: Sequence[str],
    config: SpotterConfig,
    rnd: Optional[random.Random] = None,
) -> List[str]:
    """
    Simulate the spotter output for a single document (sequence of true gloss tokens).

    Behavior:
      - If a true token is *not* in spotter_vocab -> it's unrecognizable -> treated as false-negative with probability 1.
      - Otherwise, for each true token:
           * with per-gloss FN probability (if provided) or global p_false_negative -> drop (false negative).
           * else with p_substitution -> replace with another token from spotter_vocab (substitution).
           * else keep the token (recognized correctly).
      - After processing true tokens, insert spurious tokens with probability p_false_positive per true-token position.
    """
    rnd_local = rnd or random.Random(config.seed)
    spotted: List[str] = []
    spot_vocab = list(config.spotter_vocab)

    # Precompute per-gloss rates
    per_fn = config.per_gloss_fn or {}
    per_fp = config.per_gloss_fp or {}

    for tok in true_tokens:
        # If token isn't in spotter's vocabulary, it cannot be recognized unless substituted into a known token.
        if tok not in spot_vocab:
            # treat as FN (drop) unless substitution occurs
            p_fn_effective = 1.0  # default drop if not in vocab
        else:
            p_fn_effective = float(per_fn.get(tok, config.p_false_negative))

        # false negative
        if rnd_local.random() < p_fn_effective:
            # dropped - nothing appended for this true token
            pass
        else:
            # recognized (either correct or substitution)
            if rnd_local.random() < config.p_substitution:
                # substitution: pick a different token from spotter vocab (avoid same if possible)
                if len(spot_vocab) <= 1:
                    chosen = spot_vocab[0]
                else:
                    # avoid choosing the same gloss if present
                    candidates = [g for g in spot_vocab if g != tok]
                    chosen = (
                        rnd_local.choice(candidates) if candidates else spot_vocab[0]
                    )
                spotted.append(chosen)
            else:
                # correctly recognized token
                if tok in spot_vocab:
                    spotted.append(tok)
                else:
                    # fallback: choose a random spotter token (unlikely to happen because p_fn_effective would be 1)
                    spotted.append(rnd_local.choice(spot_vocab))

        # possible insertion (false positive) after this position
        p_fp_effective = float(per_fp.get(tok, config.p_false_positive))
        if rnd_local.random() < p_fp_effective:
            inserted = rnd_local.choice(spot_vocab)
            spotted.append(inserted)

    return spotted


# -------------------------
# Length estimator simulation
# -------------------------


def estimate_length(
    true_tokens: Sequence[str],
    config: LengthEstimatorConfig,
    rnd: Optional[random.Random] = None,
) -> int:
    """Estimate document length from noisy segmenter."""
    rnd_local = rnd or random.Random(config.seed)
    true_len = len(true_tokens)
    noise = rnd_local.gauss(0.0, config.sigma)
    estimated = int(round(true_len + config.bias + noise))
    if estimated < config.min_len:
        estimated = config.min_len
    if config.max_len is not None and estimated > config.max_len:
        estimated = config.max_len
    return estimated


# -------------------------
# Corpus I/O
# -------------------------
def write_corrupted_corpus(
    corrupted_docs: List[Tuple[str, Sequence[str], Sequence[str], int]],
    out_jsonl: Path,
    use_length_estimator: bool,
) -> None:
    """
    Write JSONL with doc_id, true tokens, spotted tokens, and both length measures.
    """
    with out_jsonl.open("w", encoding="utf-8") as f:
        for doc_id, true_tokens, spotted_tokens, est_len, length_used in corrupted_docs:
            spotted_len = len(spotted_tokens)
            rec = {
                "doc_id": doc_id,
                "true_tokens": true_tokens,
                "spotted_tokens": spotted_tokens,
                "length_true": len(true_tokens),
                "length_spotted": spotted_len,
                "length_estimated": est_len,
                "length_used": length_used,
                "length_source": "estimator" if use_length_estimator else "spotted",
            }
            f.write(json.dumps(rec) + "\n")


def load_corpus_jsonl(path: Path) -> List[Tuple[str, List[str]]]:
    """
    Load a JSONL corpus where each line is {"doc_id": "...", "tokens": ["g1","g2",...]}
    or {"doc_id": "...", "true_tokens": [...]}.
    """
    docs: List[Tuple[str, List[str]]] = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "true_tokens" in rec:
                toks = list(rec["true_tokens"])
            elif "tokens" in rec:
                toks = list(rec["tokens"])
            elif "transcript" in rec:
                toks = rec["transcript"].split()
            elif "text" in rec:
                toks = rec["text"].split()
            else:
                raise ValueError(
                    "input JSONL must have 'true_tokens' or 'tokens' field"
                )

            docs.append((str(rec.get("doc_id", f"doc_{len(docs)}")), toks))
    return docs


# -------------------------
# Tiny BM25 utilities
# -------------------------
def build_bm25_index(docs: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """
    Build a simple BM25 index from a sequence of precomputed JSONL-style records.

    Each record in `docs` should be a dict with the following fields:
      - 'doc_id': str
          Unique document identifier.
      - 'spotted_tokens': Sequence[str]
          Tokenized document content (can be true tokens or spotted tokens).
      - 'length_used': int
          Precomputed document length to use for BM25 (either estimated length or
          actual token count).

    The index contains:
      - N: total number of documents
      - df: document frequency per token
      - idf: inverse document frequency per token
      - tf: term frequency per document (dict of token -> count)
      - doc_len: length used per document (from 'length_used')
      - avgdl: average document length across all documents

    Returns:
        Dict[str, object]: BM25 index dictionary suitable for `bm25_score`.
    """
    N = len(docs)
    df: Dict[str, int] = {}
    tf: Dict[str, Dict[str, int]] = {}
    doc_len: Dict[str, int] = {}

    for rec in docs:
        doc_id = rec["doc_id"]
        tokens: List[str] = rec["spotted_tokens"]
        length_used: int = rec["length_used"]

        seen: Dict[str, int] = {}
        for t in tokens:
            seen[t] = seen.get(t, 0) + 1
        tf[doc_id] = seen
        doc_len[doc_id] = length_used

        for t in seen:
            df[t] = df.get(t, 0) + 1

    avgdl = mean(doc_len.values()) if doc_len else 0.0
    idf: Dict[str, float] = {
        t: math.log(1 + (N - dfv + 0.5) / (dfv + 0.5)) for t, dfv in df.items()
    }

    return {"N": N, "df": df, "idf": idf, "tf": tf, "doc_len": doc_len, "avgdl": avgdl}



def bm25_score(
    query: Sequence[str],
    index: Dict[str, object],
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[str, float]:
    """
    Score each document in the index for the given query (bag-of-terms BM25).
    Returns mapping doc_id -> score.
    """
    idf: Dict[str, float] = index["idf"]
    tf_index: Dict[str, Dict[str, int]] = index["tf"]
    doc_len: Dict[str, int] = index["doc_len"]
    avgdl: float = index["avgdl"]
    scores: Dict[str, float] = {}
    for doc_id, tfs in tf_index.items():
        s = 0.0
        dl = doc_len.get(doc_id, 0)
        for q in query:
            qidf = idf.get(q, 0.0)
            fq = tfs.get(q, 0)
            denom = fq + k1 * (1 - b + b * (dl / avgdl) if avgdl > 0 else 1)
            if denom > 0:
                s += qidf * ((fq * (k1 + 1)) / denom)
        scores[doc_id] = s
    return scores


# ------------------------
# SIgn Spotting Metrics
# ------------------------
def sign_spotting_precision_recall_at_k(
    true_tokens: Sequence[str],
    retrieved_tokens: Sequence[str],
    k: int
) -> Tuple[float, float]:
    """
    Compute Precision@k and Recall@k for a single document.

    Args:
        true_tokens: the ground-truth tokens for this document
        retrieved_tokens: the tokens retrieved / spotted by the model
        k: cutoff for top-k tokens

    Returns:
        precision, recall: both in [0, 1]
    """
    if not true_tokens:
        return 0.0, 0.0

    top_k = retrieved_tokens[:k]
    correct = sum(1 for t in top_k if t in true_tokens)

    precision = correct / min(k, len(top_k))
    recall = correct / len(true_tokens)

    return precision, recall


# --------------------------
# Retrieval Performance
# --------------------------
def bm25_score_degradation(
    scores_true: Dict[str, float],
    scores_spotted: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute relative drop in BM25 score per document.
    """
    degradation: Dict[str, float] = {}
    for doc_id, s_true in scores_true.items():
        s_spot = scores_spotted.get(doc_id, 0.0)
        degradation[doc_id] = 1.0 - (s_spot / s_true) if s_true > 0 else 0.0
    return degradation

def rank_correlation_at_k(
    true_ranking: List[str],  # doc_ids sorted by true BM25
    spotted_ranking: List[str],  # doc_ids sorted by spotted BM25
    k: int
) -> float:
    """
    Returns fraction of overlap in top-k docs.
    """
    top_true = set(true_ranking[:k])
    top_spot = set(spotted_ranking[:k])
    return len(top_true & top_spot) / k


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg

def ndcg_at_k(
    top_docs: Sequence[str],
    scores_true: Dict[str, float],
    k: int
) -> float:
    """
    Compute NDCG@k for retrieved documents against true BM25 scores.

    Args:
        top_docs: ordered list of doc_ids retrieved by the system
        scores_true: mapping doc_id -> relevance score from true corpus
        k: cutoff

    Returns:
        NDCG@k in [0,1]
    """
    # Relevance in retrieved order
    rels = [scores_true.get(doc_id, 0.0) for doc_id in top_docs]
    dcg = dcg_at_k(rels, k)

    # Ideal DCG (sort by true relevance)
    ideal_rels = sorted(scores_true.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)

    return dcg / idcg if idcg > 0 else 0.0

# -------------------------
# Runner / CLI
# -------------------------
def run_simulation(
    *,
    input_jsonl: Optional[Path],
    synthetic: bool,
    n_docs: int,
    avg_doc_len: int,
    vocab_size: int,
    out_jsonl: Path,
    spotter_vocab_fraction: float,
    p_fn: float,
    p_fp: float,
    p_sub: float,
    use_length_estimator: bool,
    length_bias: int,
    length_sigma: float,
    seed: int,
) -> None:
    """
    Main simulation runner.
    """
    rnd = random.Random(seed)

    # Create or load corpus
    if input_jsonl:
        LOGGER.info("Loading corpus from %s", input_jsonl)
        docs_list = load_corpus_jsonl(input_jsonl)
    elif synthetic:
        LOGGER.info(
            "Generating synthetic corpus: n_docs=%d avg_doc_len=%d vocab_size=%d",
            n_docs,
            avg_doc_len,
            vocab_size,
        )
        corpus_vocab = [f"g{idx}" for idx in range(vocab_size)]
        docs_list = []
        for di in range(n_docs):
            doc_len = max(
                1, rnd.randint(max(1, avg_doc_len // 2), max(1, avg_doc_len * 3 // 2))
            )
            toks = [rnd.choice(corpus_vocab) for _ in range(doc_len)]
            docs_list.append((f"doc_{di}", toks))
    else:
        raise ValueError("Either input_jsonl must be provided or --synthetic flag used")

    # Spotter vocabulary
    corpus_vocab_set = sorted({t for _, toks in docs_list for t in toks})
    spotter_vocab_count = max(1, int(len(corpus_vocab_set) * spotter_vocab_fraction))
    spotter_vocab = rnd.sample(corpus_vocab_set, k=spotter_vocab_count)

    spot_cfg = SpotterConfig(
        spotter_vocab=spotter_vocab,
        p_false_negative=p_fn,
        p_false_positive=p_fp,
        p_substitution=p_sub,
        per_gloss_fn=None,
        per_gloss_fp=None,
        seed=seed,
    )
    len_cfg = LengthEstimatorConfig(bias=length_bias, sigma=length_sigma, seed=seed)
    LOGGER.info(f"Length Estimation config: {len_cfg}")

    LOGGER.info(
        "Spotter vocab size=%d (%.2f%% of corpus vocab=%d)",
        len(spotter_vocab),
        100.0 * spotter_vocab_fraction,
        len(corpus_vocab_set),
    )

    # Prepare output corrupted docs
    corrupted_docs: List[Tuple[str, Sequence[str], Sequence[str], int, int]] = []
    for doc_id, true_tokens in tqdm(docs_list, desc="Simulating spotter", unit="doc"):
        spotted = simulate_spotter_on_doc(true_tokens=true_tokens, config=spot_cfg, rnd=rnd)
        est_len = estimate_length(true_tokens=true_tokens, config=len_cfg, rnd=rnd)
        length_used = est_len if use_length_estimator else len(spotted)
        corrupted_docs.append((doc_id, list(true_tokens), spotted, est_len, length_used))

    # Write out corrupted corpus
    write_corrupted_corpus(corrupted_docs=corrupted_docs, out_jsonl=out_jsonl, use_length_estimator=use_length_estimator)
    LOGGER.info("Wrote corrupted corpus to %s (%d docs)", out_jsonl, len(corrupted_docs))

    # Build BM25 indices using JSONL-style dicts
    idx_true = build_bm25_index([
        {"doc_id": doc_id, "spotted_tokens": true_tokens, "length_used": len(true_tokens)}
        for doc_id, true_tokens, _, _, _ in corrupted_docs
    ])
    idx_spotted = build_bm25_index([
        {"doc_id": doc_id, "spotted_tokens": spotted, "length_used": length_used}
        for doc_id, _, spotted, _, length_used in corrupted_docs
    ])

    # Compute probes: DF and avgdl
    df_true, avgdl_true = compute_df_and_avgdl([true_tokens for _, true_tokens, _, _, _ in corrupted_docs])
    df_spotted, avgdl_spotted_tokens = compute_df_and_avgdl([spotted for _, _, spotted, _, _ in corrupted_docs])
    estimated_lengths = [length_used for _, _, _, _, length_used in corrupted_docs]
    avgdl_estimated = mean(estimated_lengths)

    LOGGER.info(
        "Corpus probes (true): vocab=%d avgdl=%.2f docs=%d",
        len(df_true), avgdl_true, len(corrupted_docs)
    )
    LOGGER.info(
        "Corpus probes (spotted tokens): vocab=%d avgdl=%.2f docs=%d",
        len(df_spotted), avgdl_spotted_tokens, len(corrupted_docs)
    )
    LOGGER.info(
        "Corpus probes (estimated lengths): avgdl=%.2f docs=%d",
        avgdl_estimated, len(corrupted_docs)
    )

    # Sample BM25 queries
    top_terms = sorted(df_true.items(), key=lambda kv: -kv[1])[:10]
    sample_queries = [[t] for (t, _) in top_terms]
    LOGGER.info("Sample queries (top true terms): %s", [q[0] for q in sample_queries])

    k=10
    for q in sample_queries:
        scores_true = bm25_score(q, idx_true)
        scores_spotted = bm25_score(q, idx_spotted)

        top_true = sorted(scores_true.items(), key=lambda kv: -kv[1])[:k]
        top_spotted = sorted(scores_spotted.items(), key=lambda kv: -kv[1])[:k]

        # LOGGER.info("Query=%s -> \n\ttop_true=%s \n\ttop_spotted=%s", q, top_true, top_spotted)

        deg = bm25_score_degradation(scores_true, scores_spotted)
        avg_deg = mean(deg.values())

        true_ids = [doc_id for doc_id, _ in top_true]
        spot_ids = [doc_id for doc_id, _ in top_spotted]
        correlation = rank_correlation_at_k(true_ids, spot_ids, k=k)

        LOGGER.info(
            "Query=%s -> Avg BM25 score degradation: %.3f. Rank Correlation@%d: %.3f",
            q,
            avg_deg,
            k,
            correlation,
        )
        







def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate sign-spotter noise and length estimation."
    )
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional input JSONL corpus (doc_id, true_tokens/tokens).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic corpus instead of loading input.",
    )
    p.add_argument(
        "--n-docs", type=int, default=1000, help="Number of synthetic docs to generate."
    )
    p.add_argument(
        "--avg-doc-len",
        type=int,
        default=20,
        help="Average document length (for synthetic corpus).",
    )
    p.add_argument(
        "--vocab-size", type=int, default=200, help="Synthetic corpus vocabulary size."
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("corrupted_corpus.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument(
        "--spotter-vocab-fraction",
        type=float,
        default=0.9,
        help="Fraction of corpus vocab present in spotter vocab.",
    )
    p.add_argument(
        "--p-fn",
        type=float,
        default=0.1,
        help="Global per-token false negative probability.",
    )
    p.add_argument(
        "--p-fp",
        type=float,
        default=0.02,
        help="Global per-token false positive (insertion) probability.",
    )
    p.add_argument(
        "--p-sub",
        type=float,
        default=0.0,
        help="Global per-token substitution probability (recognized as wrong gloss).",
    )
    p.add_argument(
        "--use-length-estimator",
        action="store_true",
        help="Whether BM25 should use the estimated lengths or count spotted tokens",
    )
    p.add_argument(
        "--length-bias",
        type=int,
        default=0,
        help="Bias added to estimated length (positive or negative).",
    )
    p.add_argument(
        "--length-sigma",
        type=float,
        default=1.0,
        help="Stddev for noise in length estimator (Gaussian).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_simulation(
        input_jsonl=args.input_jsonl,
        synthetic=args.synthetic,
        n_docs=args.n_docs,
        avg_doc_len=args.avg_doc_len,
        vocab_size=args.vocab_size,
        out_jsonl=args.out_jsonl,
        spotter_vocab_fraction=max(0.0, min(1.0, args.spotter_vocab_fraction)),
        p_fn=max(0.0, min(1.0, args.p_fn)),
        p_fp=max(0.0, min(1.0, args.p_fp)),
        p_sub=max(0.0, min(1.0, args.p_sub)),
        use_length_estimator=args.use_length_estimator,
        length_bias=args.length_bias,
        length_sigma=max(0.0, args.length_sigma),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
