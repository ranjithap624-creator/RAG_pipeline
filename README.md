# RAG_pipeline
This document provides a clear, well‑structured explanation of how to build a Retrieval-Augmented Generation (RAG) 

1. Install Required Libraries
Run the following commands in Google Colab:
!pip install langchain langchain-community
!pip install faiss-cpu
!pip install pypdf python-docx
!pip install sentence-transformers
!pip install transformers


 2. Upload a File (PDF, DOCX, or TXT)
from google.colab import files
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
print("Uploaded:", file_path)

This allows you to upload a document that the RAG system will process.

 3. Load the Document Using the Appropriate Loader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".docx"):
    loader = Docx2txtLoader(file_path)
else:
    loader = TextLoader(file_path)

docs = loader.load()

This ensures the system works with multiple file types.

4. Split Document Into Chunks
To help the model process large text, split into overlapping chunks:
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)
print(f"Total Chunks: {len(documents)}")

Overlapping chunks improve accuracy when retrieving context.

5. Create Embeddings and Store Them in FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

FAISS enables fast similarity search across vector embeddings.

6. Load FLAN‑T5 Text Generation Model
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)

llm = HuggingFacePipeline(pipeline=flan_pipeline)

FLAN‑T5 is lightweight, fast, and effective for Q&A tasks. 7. Build the RetrievalQA Chain
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

This connects the retriever (FAISS) with the language model.

 8. Quick Test Query
query = "Give me a short summary of the document"
print(qa.run(query))

This verifies that the pipeline works correctly.

9. Interactive Question‑Answering Loop
while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print("Answer:", qa.run(q))

You can now ask unlimited questions based on your uploaded document.

 Google Colab Notebook Link
Access the interactive notebook here:
 https://colab.research.google.com/drive/1VUOXDW-0QERdK-nZltOov82_0bRggBgO?usp=sharing

Summary
This RAG pipeline allows you to:
Upload documents in multiple formats


Process them into small chunks


Convert them into embeddings


Store them efficiently with FAISS


Generate accurate answers using FLAN‑T5


This is a powerful setup for creating document chatbots, knowledge‑based assistants, or search systems.
