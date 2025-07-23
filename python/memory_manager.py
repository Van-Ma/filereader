from langchain_community.vectorstores import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import uuid
import os
import psycopg2

# Load environment variables (DB connection string)
from dotenv import load_dotenv
load_dotenv()

DB_CONNECTION_STRING = os.getenv("PGVECTOR_DB_URI")  # e.g. postgresql+psycopg2://user:pass@host/dbname

COLLECTION_NAME = "chat_memory"

# Embedding function (free HuggingFace model)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize PGVector store
vectorstore = PGVector(
    connection_string=DB_CONNECTION_STRING,
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME,
)

def save_chat_memory(chat_id: str, user_prompt: str, agent_response: str):
    """
    Save a chat turn (prompt + response) into PostgreSQL vectorstore with chat_id.
    """
    combined_text = f"User: {user_prompt}\nAgent: {agent_response}"
    doc = Document(
        page_content=combined_text,
        metadata={
            "chat_id": chat_id,
            "message_id": str(uuid.uuid4())
        }
    )
    vectorstore.add_documents([doc])

def retrieve_chat_memory(chat_id: str, query: str, k: int = 5) -> List[Document]:
    """
    Retrieve relevant chat history documents by similarity for a chat_id.
    """
    return vectorstore.similarity_search(query=query, k=k, filter={"chat_id": chat_id})

def delete_chat_memory(chat_id: str):
    """
    Delete all vectors related to a chat_id via direct SQL since PGVector wrapper doesn't support deletion yet.
    """
    # Parse DB connection string for psycopg2 connection
    # Example DB_CONNECTION_STRING: postgresql+psycopg2://user:password@host:port/dbname
    import re
    pattern = r'postgresql\+psycopg2://([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/(.+)'
    match = re.match(pattern, DB_CONNECTION_STRING)
    if not match:
        raise ValueError("Invalid DB_CONNECTION_STRING format")

    user, password, host, port, dbname = match.groups()
    port = port or '5432'

    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
    try:
        with conn.cursor() as cur:
            # Delete rows where metadata->>'chat_id' = chat_id
            delete_sql = f"""
                DELETE FROM {COLLECTION_NAME}
                WHERE metadata->>'chat_id' = %s
            """
            cur.execute(delete_sql, (chat_id,))
            conn.commit()
    finally:
        conn.close()
