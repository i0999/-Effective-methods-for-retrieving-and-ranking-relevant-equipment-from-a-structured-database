import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile

# Import from your thesis.py file
from thesis import (
    load_equipment_file,
    filter_equipment_rows,
    EquipmentRecommender,
    HybridEquipmentRecommender,
)

st.set_page_config(page_title="Equipment Search", layout="wide")

st.title("Equipment Question Answering")
st.write("Upload your equipment database, type a question, and get the best matching equipment.")

# Sidebar settings
st.sidebar.header("Settings")
model_type = st.sidebar.selectbox(
    "Retriever",
    ["Hybrid (BM25 + embeddings)", "BM25 only"]
)
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)
apply_filter = st.sidebar.checkbox("Filter likely non-equipment rows", value=True)

uploaded_file = st.file_uploader(
    "Upload equipment database",
    type=["csv", "xlsx", "xls"]
)

@st.cache_resource
def build_system(file_bytes: bytes, filename: str, model_type: str, apply_filter: bool):
    """
    Build the recommender once and cache it.
    """
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    df = load_equipment_file(temp_path)

    if apply_filter:
        df = filter_equipment_rows(df)

    if model_type == "Hybrid (BM25 + embeddings)":
        system = HybridEquipmentRecommender(df)
    else:
        system = EquipmentRecommender(df)

    return system, df

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    with st.spinner("Building recommender..."):
        system, df = build_system(
            file_bytes=file_bytes,
            filename=uploaded_file.name,
            model_type=model_type,
            apply_filter=apply_filter,
        )

    st.success(f"Loaded {len(df)} rows.")

    query = st.text_input(
        "Ask your question",
        placeholder="Example: I need equipment for infrared spectroscopy of polymers"
    )

    if query:
        with st.spinner("Searching..."):
            results = system.search(query, top_k=top_k)

        st.subheader("Results")
        st.dataframe(results, width="stretch", hide_index=True)

        # Optional: show each result as a card-like block
        st.subheader("Detailed view")
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            with st.expander(f"{i}. {row.get('Equipment name', 'Unknown equipment')}"):
                st.write("**Aliases / synonyms:**", row.get("Aliases / synonyms", ""))
                st.write("**Description:**", row.get("Short description (2-3 sentences)", ""))
                st.write("**Tags:**", row.get("Tags (5-10 keywords)", ""))
                st.write("**Inputs:**", row.get("Inputs", ""))
                st.write("**Outputs:**", row.get("Outputs", ""))
                st.write("**University:**", row.get("University", ""))
                st.write("**Confidence:**", row.get("Confidence", ""))

                if "bm25_score" in row:
                    st.write("**BM25 score:**", row["bm25_score"])
                if "vector_score" in row:
                    st.write("**Vector score:**", row["vector_score"])
                if "final_score" in row:
                    st.write("**Final score:**", row["final_score"])
else:
    st.info("Upload a CSV or Excel file to begin.")