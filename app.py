
import hashlib
import re

import faiss
import numpy as np
import ollama
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
MIN_SIMILARITY = 0.25

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG Document Assistant")
st.write("Upload a PDF and ask questions grounded in its content.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(pdf_file):
    """Extract readable text while preserving PDF page numbers."""
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Normalize whitespace without joining separate words.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", "\n", text).strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def split_into_chunks(pages, chunk_size=160, overlap=30):
    """Create word-based chunks without crossing PDF page boundaries."""
    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            chunk_words = words[start:start + chunk_size]

            if chunk_words:
                chunks.append({
                    "page": page["page"],
                    "text": " ".join(chunk_words),
                })

            start += step

    return chunks


def build_search_index(chunks, model):
    """Create a FAISS index from normalized text embeddings."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(question, chunks, model, index, top_k=TOP_K):
    """Retrieve the highest-scoring chunks for a question."""
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32",
    )

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(score),
            })

    # Avoid returning the same chunk more than once.
    unique_results = []
    seen = set()

    for item in results:
        key = item["text"]

        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return unique_results


def generate_answer(question, results):
    """Generate an answer only when retrieval provides usable evidence."""
    relevant = [
        item for item in results
        if item["score"] >= MIN_SIMILARITY
    ]

    if not relevant:
        return (
            "I couldn't find sufficiently relevant information "
            "in this document to answer the question."
        )

    context = "\n\n".join(
        f"Passage {i} (PDF page {item['page']}):\n{item['text']}"
        for i, item in enumerate(relevant, start=1)
    )

    prompt = f"""
You are a careful document question-answering assistant.

Answer the question using ONLY explicit evidence in the
document passages below. Do not use outside knowledge.

Follow these rules:
1. If the evidence is missing or insufficient, say:
   "I could not find enough information in the document
   to answer this question."
2. For comparison questions, do not infer differences merely
   because two tools or topics are mentioned separately.
   The passages must support the requested comparison.
3. Do not invent facts or unsupported explanations.
4. Give a clear, concise answer.
5. Cite the supporting passage number and PDF page when possible.
6. Passage numbers refer to the passages provided below.
7. If only part of the question can be answered, clearly state
   what the document supports and what it does not establish.

DOCUMENT PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


uploaded_pdf = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_pdf:
    try:
        pdf_bytes = uploaded_pdf.getvalue()
        file_key = hashlib.sha256(pdf_bytes).hexdigest()

        with st.spinner("Reading your PDF..."):
            uploaded_pdf.seek(0)
            pages = extract_pdf_text(uploaded_pdf)

        if not pages:
            st.error(
                "No readable text was found. This may be a scanned "
                "PDF that needs OCR."
            )
            st.stop()

        # Build or rebuild the index whenever the PDF changes.
        if st.session_state.get("file_key") != file_key:
            with st.spinner("Preparing document search..."):
                chunks = split_into_chunks(pages)

                if not chunks:
                    st.error("No text chunks could be created.")
                    st.stop()

                embedding_model = load_embedding_model()
                search_index = build_search_index(
                    chunks,
                    embedding_model,
                )

                st.session_state["chunks"] = chunks
                st.session_state["search_index"] = search_index
                st.session_state["file_key"] = file_key

        chunks = st.session_state["chunks"]
        search_index = st.session_state["search_index"]
        embedding_model = load_embedding_model()

        st.success(
            f"Document ready! Extracted text from {len(pages)} pages "
            f"and created {len(chunks)} chunks."
        )

        question = st.text_input(
            "Ask a question about your document",
            placeholder="e.g. What is a Chef recipe?",
        )

        if st.button("Get Answer", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    with st.spinner("Searching the document..."):
                        results = retrieve_chunks(
                            question,
                            chunks,
                            embedding_model,
                            search_index,
                        )

                    st.subheader("Answer")

                    if not results:
                        st.warning(
                            "No passages could be retrieved from "
                            "the document."
                        )

                    else:
                        with st.spinner("Generating a grounded answer..."):
                            answer = generate_answer(
                                question,
                                results,
                            )

                        st.write(answer)

                    with st.expander(
                        "View retrieved passages and scores"
                    ):
                        for i, item in enumerate(results, start=1):
                            st.markdown(
                                f"**Passage {i} — PDF page {item['page']}**"
                            )
                            st.caption(
                                "Similarity score: "
                                f"{item['score']:.3f}"
                            )
                            st.write(item["text"])

                except Exception as error:
                    st.error(
                        "Could not generate the answer. Check that "
                        "Ollama is running and the required model "
                        "is installed."
                    )
                    st.caption(f"Technical details: {error}")

    except Exception as error:
        st.error(
            "Something went wrong while reading or preparing "
            "the PDF. Check that it contains readable text."
        )
        st.caption(f"Technical details: {error}")
