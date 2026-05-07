import os
import json
import ssl
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
load_dotenv()

# --- הגדרת אירועים ---
class RetrievalEvent(Event):
    context: str
    query: str
    is_structured: bool = False

# --- ה-Workflow המשודרג (שלב ג') ---
class RAGWorkflow(Workflow):
    def __init__(self, index, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.llm = llm

    @step
    async def router(self, ev: StartEvent) -> RetrievalEvent | StopEvent:
        query = ev.query
        if not query: return StopEvent(result="לא התקבלה שאלה.")

        print(f"🚦 נתב: מנתח את סוג השאלה: {query}")
        
        # זיהוי האם השאלה דורשת רשימה מובנית
        list_keywords = ["רשימה", "החלטות", "חוקים", "אזהרות", "כל ההחלטות", "כל הכללים", "סכם"]
        is_list_query = any(word in query for word in list_keywords)

        if is_list_query:
            print("📊 נתיב מובנה: שולף נתונים מ-JSON")
            try:
                with open("extracted_data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                context = json.dumps(data, ensure_ascii=False, indent=2)
                return RetrievalEvent(context=context, query=query, is_structured=True)
            except:
                print("⚠️ קובץ JSON לא נמצא, עובר לחיפוש סמנטי")

        # חיפוש סמנטי רגיל (שלב א'+ב')
        print("🔍 נתיב סמנטי: מחפש בוקטורים")
        retriever = self.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        context = "\n".join([n.get_content() for n in nodes])
        return RetrievalEvent(context=context, query=query, is_structured=False)

    @step
    async def generate(self, ev: RetrievalEvent) -> StopEvent:
        print(f"🤖 מנסח תשובה סופית (Structured: {ev.is_structured})")
        
        if ev.is_structured:
            prompt = (
                f"המשתמש ביקש רשימה מובנית. השאלה: {ev.query}\n"
                f"הנה הנתונים המובנים מהפרויקט:\n{ev.context}\n\n"
                "ארגן את התשובה בצורה של רשימה מסודרת וברורה לפי סוגים (החלטות, חוקים, אזהרות)."
            )
        else:
            prompt = (
                f"ענה על השאלה: {ev.query}\n"
                f"בהתבסס על המידע הבא:\n{ev.context}\n\n"
                "אם המידע לא נמצא, ציין זאת."
            )
            
        response = self.llm.complete(prompt)
        return StopEvent(result=str(response))

# --- טעינת מערכת ---
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

async def chat_response(message, history):
    workflow = RAGWorkflow(index=index, llm=llm, timeout=60)
    result = await workflow.run(query=message)
    return result

demo = gr.ChatInterface(fn=chat_response, title="סוכן AI משולב (RAG + Extraction)")

if __name__ == "__main__":
    print("🎨 מנסה לייצר תרשים זרימה...")
    try:
        # יצירת התרשים
        draw_all_possible_flows(RAGWorkflow, filename="workflow_graph.html")
        
        # בדיקה אם הקובץ באמת נוצר
        if os.path.exists("workflow_graph.html"):
            full_path = os.path.abspath("workflow_graph.html")
            print(f"✅ הצלחתי! התרשים נוצר בכתובת:\n{full_path}")
        else:
            print("❌ הקוד רץ אבל הקובץ לא נמצא בתיקייה.")
    except Exception as e:
        print(f"⚠️ שגיאה ביצירת התרשים: {e}")

    print("🚀 מפעיל את ממשק Gradio...")
    demo.launch()