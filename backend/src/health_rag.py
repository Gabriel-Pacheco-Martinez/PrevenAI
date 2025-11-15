"""
This scripts reads documents, chunks them, creates embeddings, and stores 
them in a vector database (FAISS) for retrieval-augmented generation (RAG).
Qwen2.5-7b + LoRA adapter for domain-specific QA
"""

# General
import os
import torch

# Huggingface
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader
)


class HealthRag:
    # =======
    # Initialize RAG system: only once
    # =======
    def __init__(self, docs_path="docs/", base_model_hf="Qwen/Qwen2.5-7B",
                 lora_repo="gabopachecoo2000/qwen2.5-7b-lora-customization-pipeline",
                 faiss_path="faiss", chunk_size=700, chunk_overlap=150):
        
        print("\n========== INITIALIZING RAG SYSTEM ==========\n")

        # ====
        # Load base LLM with LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_hf,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print(f"[✓] 1.- Loaded base model correctly")

        # Wrap base model with LoRA adapter weights
        self.peft_model = PeftModel.from_pretrained(base_model, lora_repo)
        self.peft_model.eval()
        print(f"[✓] 2.- Loaded LoRA adapter correctly")

        # Tokenizer object to translate text ↔ token IDs when doing RAG queries
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_hf)

        # ====
        # Load documents → chunks
        docs = self.load_documents(docs_path)
        print(f"[✓] 3.- Loaded {len(docs)} documents")

        # Turn into chunks
        chunks = self.chunk_documents(docs, chunk_size, chunk_overlap)
        print(f"[✓] 4.- Chunked documents")

        # Load embedder
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Store vectors in FAISS
        if os.path.exists(faiss_path):
            print("    ↳ Loading existing FAISS folder and index...")
            self.faiss_db = FAISS.load_local(faiss_path, self.embedder)
        else:
            print("    ↳ Creating new FAISS folder and index...")
            texts = [c.page_content for c in chunks]
            self.faiss_db = FAISS.from_texts(texts, self.embedder)
            self.faiss_db.save_local(faiss_path)
        print("[✓] 5.- FAISS index saved.")

        print("\n✅ ========== RAG SYSTEM READY ==========\n")


    # ============================================
    # HELPERS
    # ============================================
    def load_documents(self, path):
        docs = []
        for file in os.listdir(path):
            full_path = os.path.join(path, file)

            if file.endswith(".pdf"):
                docs.extend(PyPDFLoader(full_path).load())
            elif file.endswith(".docx"):
                docs.extend(Docx2txtLoader(full_path).load())
            elif file.endswith(".pptx"):
                docs.extend(UnstructuredPowerPointLoader(full_path).load())
            elif file.endswith(".txt"):
                docs.extend(TextLoader(full_path).load())
            else:
                print(f"[Skipping] Unsupported file: {file}")

        print(f"    ↳ Loaded {len(docs)} documents")
        return docs


    def chunk_documents(self, documents, chunk_size, chunk_overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(documents)


    # ============================================
    # RAG QUERY FUNCTION
    # ============================================
    def ask_enhanced_llm(self, query, k=5):
        # Retrieve relevant chunks
        docs = self.faiss_db.similarity_search(query, k=k)
        context = "\n\n".join([d.page_content for d in docs])

        # Build input prompt
        prompt = f"""
You are a preventative health assistant. Use ONLY the provided context.
Give general, safe wellness advice.

Context:
{context}

User:
{query}

Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the output
        return response.split("Answer:")[-1].strip()