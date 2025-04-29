# DeepMind Research Paper Q&A System

This project implements a Question-Answering (QnA) pipeline over 855 DeepMind research papers in PDF format. It uses LangChain with ChromaDB for vector search and Google's Gemini API for answering questions.

## NOTE:download the ipynb file and see the code
## Features

- Loads PDFs in batches (10 at a time) from a local folder
- Converts PDFs to text using `unstructured` and `PyMuPDF`
- Embeds and stores documents in ChromaDB
- Answers user queries using Google's Gemini (Generative AI)

## Tech Stack

- LangChain
- ChromaDB
- SentenceTransformers
- Google Gemini API
- PyMuPDF
- Streamlit (optional for UI)

## Installation

```bash
pip install langchain langchain_community streamlit langchain_experimental
pip install sentence-transformers langchain_chroma langchainhub
pip install unstructured pymupdf tiktoken chromadb google-generativeai
pip install langchain-google-genai python-dotenv
```

## Folder Structure

```
project/
│
├── pdf_files/               # Folder containing all 855 research PDFs
├── embeddings/              # Stores ChromaDB persistent data
├── .env                     # Environment variables with Gemini API key
└── deepmind_QnA.ipynb       # Main Jupyter notebook
```

## Example Usage

```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI

# Load and split PDF
loader = PyMuPDFLoader("pdf_files/sample.pdf")
documents = loader.load()

# Embed documents
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding, persist_directory="embeddings")
db.persist()

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Ask a question
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("What is reinforcement learning?")
response = llm.invoke(docs)

print(response.content)
```

## Notes

- Set your Gemini API key in a `.env` file:
  ```
  GOOGLE_API_KEY=your_api_key_here
  ```
- Optimized for use in Colab or local machines with >=16GB RAM.
- Ideal for research summarization and academic Q&A applications.

