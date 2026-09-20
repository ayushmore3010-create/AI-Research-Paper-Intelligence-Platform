import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import plotly.express as px
import streamlit as st

from app.container import ingestion_service, metadata_store, rag_service
from app.services.extraction import summarize_paper
"""Streamlit UI for the AI Research Paper Intelligence Platform."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.container import ingestion_service, metadata_store, rag_service
from app.services.extraction import summarize_paper

st.set_page_config(page_title="Research Intelligence", page_icon="R", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
.hero {padding: 1.6rem 2rem; border-radius: 14px; background: linear-gradient(115deg,#102a43,#1d4e89); color: white; margin-bottom: 1.4rem;}
.hero h1 {font-size: 2.4rem; margin-bottom: .3rem;}
[data-testid="stMetric"] {border: 1px solid #d9e2ec; padding: 1rem; border-radius: 10px; background: #fff;}
</style>""", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>Research Intelligence</h1><p>Search, synthesize, and compare your paper library with grounded evidence.</p></div>", unsafe_allow_html=True)
papers = metadata_store.list_documents()
pages = sum(paper.page_count for paper in papers)
chunks = sum(paper.chunk_count for paper in papers)
queries = metadata_store.query_count()
metrics = st.columns(4)
metrics[0].metric("Papers", len(papers))
metrics[1].metric("Processed pages", pages)
metrics[2].metric("Indexed chunks", chunks)
metrics[3].metric("Questions asked", queries)

with st.sidebar:
    st.header("Workspace")
    section = st.radio("Navigate", ["Dashboard", "Upload papers", "My papers", "Semantic search", "Ask assistant", "Paper summary", "Compare papers", "Analytics", "Evaluation"])

if section == "Upload papers":
    st.subheader("Add research papers")
    files = st.file_uploader("Choose PDF papers", type="pdf", accept_multiple_files=True)
    if files and st.button("Process papers", type="primary"):
        for file in files:
            try:
                record = ingestion_service.ingest(file.name, file.getvalue())
                st.success(f"{record.filename} is ready ({record.chunk_count} chunks).")
            except Exception as exc:
                st.error(str(exc))
elif section == "My papers":
    st.subheader("My research library")
    for paper in papers:
        with st.expander(f"{paper.metadata.title}  |  {paper.status}"):
            st.write(f"**File:** {paper.filename}  |  **Pages:** {paper.page_count}  |  **Chunks:** {paper.chunk_count}")
            if st.button("Delete", key=f"delete-{paper.id}"):
                ingestion_service.delete(paper.id)
                st.rerun()
elif section in {"Semantic search", "Ask assistant"}:
    st.subheader(section)
    question = st.text_area("Research question", placeholder="What problem does this paper address?")
    top_k = st.slider("Sources to retrieve", 1, 10, 5)
    if st.button("Run", type="primary") and question:
        if section == "Semantic search":
            results = rag_service.search(question, top_k)
            for result in results:
                st.markdown(f"**{result.chunk.source}**  ·  relevance `{result.score:.3f}`")
                st.write(result.chunk.text)
        else:
            response = rag_service.answer(question, top_k)
            st.write(response.answer)
            st.caption("Sources")
            for citation in response.citations:
                st.info(f"{citation.document_name}, page {citation.page_number} · {citation.excerpt}")
elif section == "Paper summary":
    st.subheader("Structured paper summary")
    if papers:
        selected = st.selectbox("Paper", papers, format_func=lambda paper: paper.metadata.title)
        st.json(summarize_paper(selected.metadata))
    else:
        st.info("Upload a paper to generate a summary.")
elif section == "Compare papers":
    st.subheader("Compare your evidence")
    selected = st.multiselect("Select papers", papers, format_func=lambda paper: paper.metadata.title)
    if selected:
        rows = [{"Paper": paper.metadata.title, "Problem": paper.metadata.research_problem or "Not extracted", "Methodology": paper.metadata.methodology or "Not extracted", "Dataset": paper.metadata.dataset or "Not extracted"} for paper in selected]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
elif section in {"Dashboard", "Analytics"}:
    st.subheader("Research analytics")
    if papers:
        years = pd.DataFrame([{"Year": paper.metadata.year or "Unknown", "Paper": paper.metadata.title} for paper in papers])
        st.plotly_chart(px.histogram(years, x="Year", title="Papers by publication year"), use_container_width=True)
    else:
        st.info("Your analytics will appear after papers are indexed.")
else:
    st.subheader("RAG evaluation")
    st.write("The evaluation suite measures retrieval relevance, context precision/recall, faithfulness, and citation correctness against labeled source chunks.")
    st.info("Run `pytest` to execute the local evaluation and regression tests.")
