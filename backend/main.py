# main.py
# This is our FastAPI server — the waiter that takes requests
# and passes them to our RAG pipeline

import os
import uuid
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import tempfile

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create the FastAPI app
# Think of this as turning on the restaurant
app = FastAPI(title="DevDocs AI", version="1.0.0")

# CORS — this allows our React frontend (running on a different port)
# to talk to this backend. Without this, browsers block cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React runs on port 3000
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB client — persistent so data survives restarts
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# In-memory store to track uploaded document names
# { "doc_id": "filename.pdf" }
documents_store = {}

# ── Embedding function (same as rag_pipeline.py) ──────────────────────────────
class GeminiEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=input,
        )
        return result["embedding"] if isinstance(input, str) else [r for r in result["embedding"]]

# ── Request/Response models ───────────────────────────────────────────────────
# Pydantic models define the shape of data coming in and going out
# FastAPI uses these to validate requests automatically

class QueryRequest(BaseModel):
    # This defines what a query request must look like:
    # { "doc_id": "abc123", "question": "What are the skills?" }
    doc_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str

# ── Helper functions ──────────────────────────────────────────────────────────
def process_pdf(file_path: str, doc_id: str):
    """Load PDF, chunk it, embed it, store in ChromaDB"""
    
    # Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)
    
    # Create a separate ChromaDB collection for each document
    # This way different documents don't mix together
    try:
        chroma_client.delete_collection(doc_id)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=doc_id,
        embedding_function=GeminiEmbeddings()
    )
    
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return len(chunks), len(pages)

def get_answer(doc_id: str, question: str):
    """Retrieve relevant chunks and get answer from Gemini"""
    
    # Get the collection for this document
    try:
        collection = chroma_client.get_collection(
            name=doc_id,
            embedding_function=GeminiEmbeddings()
        )
    except:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Find relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=6
    )
    
    # Build context
    context = "\n\n".join(results["documents"][0])
    sources = [doc[:150] + "..." for doc in results["documents"][0]]
    
    # Ask Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    If the answer is not in the context, say "I don't know based on the provided document."
    
    Context: {context}
    
    Question: {question}
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, sources

# ── API Endpoints ─────────────────────────────────────────────────────────────

# GET / — health check
# Visit http://localhost:8000 to confirm the server is running
@app.get("/")
def root():
    return {"message": "DevDocs AI is running"}

# POST /upload — accepts a PDF file
# @app.post means this endpoint only responds to POST requests
# UploadFile is FastAPI's way of handling file uploads
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    
    # Validate file type — only PDFs allowed
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate a unique ID for this document
    # uuid4() creates a random unique string like "a3f2c1d4-..."
    doc_id = str(uuid.uuid4())[:8]  # We take first 8 chars to keep it short
    
    # Save the uploaded file temporarily to disk
    # We need a real file path because PyPDFLoader needs to read from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()  # await = wait for the file to fully upload
        tmp.write(content)
        tmp_path = tmp.name
    
    # Process the PDF through our RAG pipeline
    chunks_count, pages_count = process_pdf(tmp_path, doc_id)
    
    # Clean up the temp file
    os.unlink(tmp_path)
    
    # Store the document info
    documents_store[doc_id] = file.filename
    
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": pages_count,
        "chunks": chunks_count,
        "message": f"Successfully processed {pages_count} pages into {chunks_count} chunks"
    }

# POST /query — accepts a question and returns an answer
@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer, sources = get_answer(request.doc_id, request.question)
    
    return QueryResponse(answer=answer, sources=sources)

# GET /documents — returns list of all uploaded documents
@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return [
        DocumentInfo(doc_id=doc_id, filename=filename)
        for doc_id, filename in documents_store.items()
    ]