import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (.env should contain GEMINI_API_KEY)
load_dotenv()



def build_rag_pipeline(documents: list[Document]):
    # 1. Recursive Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs_split = text_splitter.split_documents(documents)
    print(f"[*] Split documents into {len(docs_split)} chunks.")

    # 2. Initialize Local HuggingFace Embeddings (Bypasses Google API model version 404 errors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Store Embeddings Locally in ChromaDB (with persistence)
    persist_directory = "./chroma_db"
    vector_store = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("[*] Vector store created and persisted locally at:", persist_directory)

    # 4. Create a Retriever from the Vector Store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 5. Initialize the LLM (Gemini 2.5 Flash for answering questions)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.1)

    # 6. Define Prompt & Chains
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

if __name__ == "__main__":
    # For demonstration, let's create a temporary sample text file to load.
    
    sample_text_content = """
    Company Policy Manual - Acme Corporation (2026 Update)
    
    Section 1: Remote Work Policy
    Employees are permitted to work remotely up to 3 days per week with prior manager approval. 
    Core working hours are between 10:00 AM and 4:00 PM local time. All remote employees must maintain 
    a secure internet connection and use company-issued VPN software. (Source Page: 12)
    
    Section 2: Travel & Expense Reimbursement
    Business travel must be booked through the corporate travel portal at least two weeks in advance. 
    Flights must be economy class. Daily meal allowances are capped at $75 per day within domestic regions 
    and $110 per day for international travel. Receipts must be uploaded within 14 days of trip completion. 
    (Source Page: 28)
    
    Section 3: Professional Development Stipend
    Every full-time employee receives an annual professional development stipend of $1,500. This can be 
    used for online courses, certifications, books, and technology conferences directly related to their role. 
    Unused stipends do not roll over into the next calendar year. (Source Page: 45)
    """

    # Write sample text to a file to simulate loading a custom document
    os.makedirs("./data", exist_ok=True)
    sample_file_path = "./data/company_policy.txt"
    with open(sample_file_path, "w") as f:
        f.write(sample_text_content)

    # Load the document using LangChain TextLoader
    print("--- Loading Document ---")
    loader = TextLoader(sample_file_path)
    raw_documents = loader.load()

    # Build the RAG system
    print("--- Building RAG Pipeline & Vector Embeddings ---")
    rag_pipeline = build_rag_pipeline(raw_documents)

    # Query the RAG system
    query = "What is the daily meal allowance for international business travel, and how far in advance should travel be booked?"
    print(f"\n--- Querying RAG System ---\nQuestion: {query}")

    response = rag_pipeline.invoke({"input": query})

    print("\n[AI Answer]:")
    print(response["answer"])

    print("\n[Retrieved Source Citations / Chunks]:")
    for idx, doc in enumerate(response["context"], 1):
        print(f"\nSource {idx}:")
        print(doc.page_content)