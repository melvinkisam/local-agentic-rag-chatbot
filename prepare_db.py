from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import glob

load_dotenv()

# 1. init embedding model
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

# 2. reset DB
db_path = os.getenv("DATABASE_LOCATION")

if os.path.exists(db_path):
    shutil.rmtree(db_path)

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=db_path
)

# 3. text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# 4. load text files from data directory
for path in glob.glob("input_data/*.txt"):
    with open(path) as f:
        raw = f.read()

    # 5. split into chunks
    docs = text_splitter.create_documents([raw], metadatas=[{"source": path}])

    # 6. add to vector store
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=ids)

    print(f"✅ Added {len(docs)} documents from {path}")

print("Done!")
