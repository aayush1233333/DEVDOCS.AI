# rag_pipeline.py - Clean version using direct Google API
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ── Custom embedding function that calls Google directly ──────────────────────
# This bypasses LangChain's broken wrapper entirely
class GeminiEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=input,
        )
        return result["embedding"] if isinstance(input, str) else [r for r in result["embedding"]]

# ── Load PDF ──────────────────────────────────────────────────────────────────
def load_pdf(file_path):
    print(f"\n📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages")
    return pages

# ── Split into chunks ─────────────────────────────────────────────────────────
def split_into_chunks(pages):
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ── Store in ChromaDB ─────────────────────────────────────────────────────────
def store_in_chromadb(chunks):
    print("\n🧠 Storing in ChromaDB...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete old collection if it exists so we start fresh
    try:
        client.delete_collection("documents")
    except:
        pass
    
    collection = client.create_collection(
        name="documents",
        embedding_function=GeminiEmbeddings()
    )
    
    # Add all chunks to ChromaDB
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return collection

# ── Ask a question ────────────────────────────────────────────────────────────
def ask_question(collection, question):
    print(f"\n❓ Question: {question}")
    
    # Find the 3 most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    
    # Build context from retrieved chunks
    context = "\n\n".join(results["documents"][0])
    
    # Ask Gemini using the context
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    If the answer is not in the context, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\n💬 Answer: {answer}")
    print(f"\n📚 Sources used:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        page = meta.get("page", 0) + 1
        print(f"  Source {i+1} (Page {page}): '{doc[:150]}...'")
    
    return answer

# ── Run everything ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pages = load_pdf("test.pdf")
    chunks = split_into_chunks(pages)
    collection = store_in_chromadb(chunks)
    
    ask_question(collection, "What are the main skills mentioned?")
    ask_question(collection, "What projects are described?")