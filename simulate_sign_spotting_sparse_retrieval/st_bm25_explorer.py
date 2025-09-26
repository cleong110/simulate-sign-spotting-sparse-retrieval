import json
import logging
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from simulate_sign_spotting_sparse_retrieval.simulate_sign_spotter import (
    bm25_score,
    build_bm25_index,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Load data once and build BM25 index
# ---------------------------------------------------------------------
import json
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

import streamlit as st

from simulate_sign_spotting_sparse_retrieval.simulate_sign_spotter import (
    bm25_score,
    build_bm25_index,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_index(data_path: Path):
    """Load transcripts JSONL and build BM25 index."""
    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            transcript: str = obj["transcript"]

            # naive tokenization: split on spaces
            tokens = transcript.split()
            rec = {
                "doc_id": doc_id,
                "spotted_tokens": tokens,
                "length_used": len(tokens),
                "transcript": transcript,  # keep transcript for display
            }
            records.append(rec)

    logger.info("Loaded %d documents", len(records))

    index = build_bm25_index(records)

    # also store transcripts inside index for UI
    index["docs"] = {rec["doc_id"]: rec["transcript"] for rec in records}
    return index



DATA_PATH = Path("simulate_sign_spotting_sparse_retrieval/test_data/phoenix2014_multisigner_segment_transcripts.jsonl")
index = load_index(DATA_PATH)

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="BM25 Explorer", layout="wide")
st.title("🔎 BM25 Explorer with Phoenix 2014 Multisigner Transcripts")

st.markdown("Type a query, adjust **k1** and **b**, and see top-scoring documents.")

query_str = st.text_input("Enter query terms (space separated):", value="MEER IX TIEF WOLKE WOLKE MORGEN IX WETTER WECHSELHAFT KALT ZWISCHEN ZWISCHEN TEMPERATUR TEMPERATUR")
query: Sequence[str] = query_str.split()

col1, col2 = st.columns(2)
with col1:
    k1 = st.slider("k1", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
with col2:
    b = st.slider("b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

if st.button("Run BM25"):
    scores = bm25_score(query=query, index=index, k1=k1, b=b)
    # Sort results
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    st.subheader("Top Results")
    for rank, (doc_id, score) in enumerate(sorted_scores[:10], start=1):
        transcript = index["docs"][doc_id]
        with st.expander(f"#{rank}: {doc_id} (score={score:.4f})"):
            st.write(transcript)
