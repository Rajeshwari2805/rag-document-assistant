# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive answers grounded in retrieved passages.

## Features

- Upload PDF documents.
- Extract text from PDF pages.
- Split document text into chunks.
- Retrieve relevant passages using semantic similarity.
- Generate document-grounded answers using a local language model.
- Display retrieved passages, PDF page numbers, and similarity scores.
- Indicate when the document does not contain enough evidence to answer a question.

## Technologies Used

- Python
- Streamlit
- Ollama
- Llama 3.2 (3B)
- Sentence Transformers
- FAISS
- PyPDF

## Requirements

- Python
- [Ollama](https://ollama.com/)
- The `llama3.2:3b` model

## Installation

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the language model:

   ```bash
   ollama pull llama3.2:3b
   ```

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## How to Use

1. Start the application.
2. Upload a PDF document.
3. Enter a question about the document.
4. Review the generated answer and the retrieved passages with their page numbers and similarity scores.

## Note

Answers depend on the information available in the uploaded document. The application may decline to answer when it cannot retrieve sufficient supporting evidence.