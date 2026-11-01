import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE =1000
CHUNK_OVERLAP =200
VECTOR_DB_PATH = "./vector_db"
