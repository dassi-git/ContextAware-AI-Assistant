import os
import ssl
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# מעקף SSL הכרחי לסביבת נטפרי/ווינדוס
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

load_dotenv()

def run_ingestion():
    print("🚀 מתחיל תהליך אינדוקס (פייתון 3.11)...")
    
    # 1. טעינת מסמכים
    path = "./my_web_api/data_source"
    if not os.path.exists(path):
        print(f"❌ שגיאה: הנתיב {path} לא נמצא!")
        return

    reader = SimpleDirectoryReader(input_dir=path, recursive=True, exclude_hidden=False)
    documents = reader.load_data()
    print(f"✅ נטענו {len(documents)} מסמכים.")

    # 2. הגדרת מודלים של OpenAI
    # ודאי שיש לך OPENAI_API_KEY בתוך קובץ ה-.env
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-3.5-turbo")

    # 3. יצירת האינדקס ושמירה מקומית
    print("📦 מעבד נתונים ושומר לתיקייה מקומית (storage)...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)]
    )

    # שמירה לדיסק
    index.storage_context.persist(persist_dir="./storage")
    print("⭐⭐⭐ הצלחנו! המידע מוכן בתיקיית storage ⭐⭐⭐")

if __name__ == "__main__":
    run_ingestion()