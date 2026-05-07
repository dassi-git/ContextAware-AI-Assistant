import os
import ssl
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# מעקף SSL לנטפרי
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
load_dotenv()

# 1. הגדרת האירוע שעובר בין השלבים
class RetrievalEvent(Event):
    context: str
    query: str

# 2. הגדרת ה-Workflow (ארכיטקטורת Event-Driven)
class RAGWorkflow(Workflow):
    def __init__(self, index, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.llm = llm

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievalEvent | StopEvent:
        query = ev.query
        if not query: return StopEvent(result="לא התקבלה שאלה.")

        print(f"🔎 סורק מסמכים עבור: {query}")
        
        # אנחנו לוקחים יותר תוצאות (5) כדי לוודא שאנחנו לא מפספסים כלום
        retriever = self.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        
        if not nodes:
            return RetrievalEvent(context="EMPTY", query=query)
        
        # אנחנו שומרים גם את שם הקובץ לכל פיסת מידע (Metadata)
        context_parts = []
        for n in nodes:
            source = n.metadata.get('file_name', 'unknown')
            context_parts.append(f"[מקור: {source}]: {n.get_content()}")
            
        context = "\n\n".join(context_parts)
        return RetrievalEvent(context=context, query=query)

    @step
    async def generate(self, ev: RetrievalEvent) -> StopEvent:
        print(f"✍️ מנסח תשובה סופית...")
        
        if ev.context == "EMPTY":
            return StopEvent(result="מצטער, לא מצאתי שום אזכור לנושא הזה במסמכי הפרויקט.")

        # פרומפט "קשוח" שמונע ממנו להמציא שטויות
        prompt = (
            f"אתה עוזר AI שמתבסס אך ורק על מסמכי הפרויקט. השאלה: {ev.query}\n\n"
            f"הנה המידע שמצאתי במסמכים:\n"
            f"{ev.context}\n\n"
            "הנחיות קריטיות:\n"
            "1. ענה אך ורק על סמך המידע שמופיע למעלה.\n"
            "2. אם המידע לא נותן תשובה ישירה לשאלה, כתוב: 'לא מצאתי מידע ספציפי על כך במסמכים'.\n"
            "3. אל תשתמש בידע כללי שלך על פוליטיקה, תאריכים או העולם.\n"
            "4. אם מצאת תשובה, ציין מאיזה קובץ לקחת אותה."
        )
        
        response = self.llm.complete(prompt)
        return StopEvent(result=str(response))

# 3. טעינת המודלים והאינדקס
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# 4. פונקציית הקישור ל-Gradio
async def chat_response(message, history):
    # יצירת מופע חדש של ה-Workflow לכל שאלה
    workflow = RAGWorkflow(index=index, llm=llm, timeout=60)
    try:
        result = await workflow.run(query=message)
        return result
    except Exception as e:
        return f"שגיאה במערכת: {e}"

# 5. ממשק המשתמש
demo = gr.ChatInterface(
    fn=chat_response, 
    title="סוכן AI מבוסס אירועים (RAG)",
    description="שאל אותי שאלות על הפרויקט שלך!"
)

if __name__ == "__main__":
    demo.launch()