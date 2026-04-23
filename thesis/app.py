import streamlit as st
from pathlib import Path
import tempfile

from thesis import (
    load_equipment_file,
    filter_equipment_rows,
    EquipmentRecommender,
    HybridEquipmentRecommender,
)

st.set_page_config(page_title="Equipment Search", layout="wide")

st.title("Equipment Question Answering")
st.write("Upload your equipment database, or use the built-in example dataset.")

st.sidebar.header("Settings")
model_type = st.sidebar.selectbox(
    "Retriever",
    ["Hybrid (BM25 + embeddings)", "BM25 only"]
)
top_k = st.sidebar.slider("Number of results", 1, 20, 5)
apply_filter = st.sidebar.checkbox("Filter likely non-equipment rows", value=True)

example_path = Path("sample_data/equipment_database.csv")

uploaded_file = st.file_uploader(
    "Upload equipment database",
    type=["csv", "xlsx", "xls"]
)

@st.cache_resource
def build_system_from_path(path_str: str, model_type: str, apply_filter: bool):
    df = load_equipment_file(path_str)

    if apply_filter:
        df = filter_equipment_rows(df)

    if model_type == "Hybrid (BM25 + embeddings)":
        system = HybridEquipmentRecommender(df)
    else:
        system = EquipmentRecommender(df)

    return system, df

@st.cache_resource
def build_system_from_upload(file_bytes: bytes, filename: str, model_type: str, apply_filter: bool):
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
    st.success(f"Using uploaded file: {uploaded_file.name}")
    system, df = build_system_from_upload(
        uploaded_file.getvalue(),
        uploaded_file.name,
        model_type,
        apply_filter,
    )
else:
    if example_path.exists():
        st.info(f"Using example dataset: {example_path}")
        system, df = build_system_from_path(
            str(example_path),
            model_type,
            apply_filter,
        )
    else:
        st.warning("No uploaded file and no example dataset found.")
        st.stop()

st.write(f"Loaded {len(df)} rows.")

query = st.text_input(
    "Ask your question",
    placeholder="Example: I need equipment for infrared spectroscopy of polymers"
)

if query:
    results = system.search(query, top_k=top_k)
    st.subheader("Results")
    st.dataframe(results, use_container_width=True)

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