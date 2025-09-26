import json
import logging
from pathlib import Path
from typing import Dict, List

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

            tokens = transcript.split()
            rec = {
                "doc_id": doc_id,
                "spotted_tokens": tokens,
                "length_used": len(tokens),
                "transcript": transcript,
            }
            records.append(rec)

    index = build_bm25_index(records)
    index["docs"] = {rec["doc_id"]: rec["transcript"] for rec in records}
    index["tokens"] = {rec["doc_id"]: rec["spotted_tokens"] for rec in records}
    return index


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------
st.set_page_config(page_title="BM25 Term Overrides", layout="wide")
st.title("🔎 BM25 Term Explorer with Overrides")

DATA_PATH = Path(
    "simulate_sign_spotting_sparse_retrieval/test_data/phoenix2014_multisigner_segment_transcripts.jsonl"
)
index = load_index(DATA_PATH)

# 1. Pick a transcript as query
doc_ids = sorted(index["docs"].keys())
selected_doc = st.selectbox("Pick a transcript as query:", options=doc_ids)

query_tokens: List[str] = index["tokens"][selected_doc]

st.markdown("**Query transcript:**")
st.write(index["docs"][selected_doc])

# 2. Compute score of the true document
col1, col2 = st.columns(2)
with col1:
    k1 = st.slider("k1", 0.0, 3.0, 1.2, 0.1)
with col2:
    b = st.slider("b", 0.0, 1.0, 0.75, 0.05)

scores = bm25_score(query=query_tokens, index=index, k1=k1, b=b)
true_score = scores[selected_doc]
st.metric("BM25 score of true transcript", f"{true_score:.4f}")

# 3. Term-level overrides
st.subheader("Term-level overrides")

dl_default = index["doc_len"][selected_doc]
avgdl_default = index["avgdl"]

doc_len = st.number_input("Document length |D|", value=dl_default, step=1)
avgdl = st.number_input("Average document length (avgdl)", value=float(avgdl_default))

idf_values: Dict[str, float] = {}
freq_values: Dict[str, int] = {}

for t in query_tokens:
    col1, col2, col3 = st.columns(3)
    with col1:
        idf_default = index["idf"].get(t, 0.0)
        idf_values[t] = st.number_input(
            f"IDF({t})", value=float(idf_default), step=0.1, key=f"idf_{t}"
        )
    with col2:
        fq_default = index["tf"][selected_doc].get(t, 0)
        freq_values[t] = st.number_input(
            f"f({t}, D)", value=int(fq_default), step=1, key=f"fq_{t}"
        )
    with col3:
        st.write(" ")  # spacing

# Recompute score with overrides
score_override = 0.0
for t in query_tokens:
    fq = freq_values[t]
    idf = idf_values[t]
    denom = fq + k1 * (1 - b + b * (doc_len / avgdl if avgdl > 0 else 1))
    if denom > 0:
        score_override += idf * ((fq * (k1 + 1)) / denom)

st.subheader("Overridden score")
st.metric("BM25 score (overrides applied)", f"{score_override:.4f}")
