from langchain_openai import OpenAIEmbeddings
try:
    from langchain_postgres import PGVector
except ImportError:
    from langchain_community.vectorstores import PGVector


import os
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env into os.environ
dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set in .env or environment")

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"))
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="methodology",
    connection=dsn,
    use_jsonb=True,
)
for d in vector_store.similarity_search("IMACEC", k=2):
    print(d.page_content[:200], d.metadata)