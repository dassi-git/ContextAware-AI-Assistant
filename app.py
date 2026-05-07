import os
import ssl
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# מעקף SSL לסביבת נטפרי
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

load_dotenv()

print("🚀 טוען את הסוכן החכם (פייתון 3.11)...")

# 1. הגדרת המודלים (חייב להתאים ל-ingest.py)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo")

# 2. טעינת הזיכרון מהתיקייה המקומית
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# 3. יצירת מנוע השאילתות
query_engine = index.as_query_engine(llm=llm)

# 4. פונקציית המענה
def chat_response(message, history):
    try:
        response = query_engine.query(message)
        return str(response)
    except Exception as e:
        return f"אופס, קרתה שגיאה: {e}"

# 5. ממשק Gradio
demo = gr.ChatInterface(
    fn=chat_response, 
    title="סוכן ה-RAG של דסי",
    description="אני מכיר את כל מסמכי הפרויקט שלך. שאל אותי כל דבר!",
    examples=["מהם צבעי המותג?", "מה כתוב ב-tech_spec?", "איזה תובנות יש ב-insights?"]
)

if __name__ == "__main__":
    demo.launch()