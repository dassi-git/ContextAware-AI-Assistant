import os
import ssl
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.llms.openai import OpenAI

# מעקף SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
load_dotenv()

# --- הגדרת הסכמה (שלב ג') ---

class ProjectItem(BaseModel):
    """פריט מידע בודד מהפרויקט"""
    type: str = Field(description="סוג הפריט: Decision, Rule, או Warning")
    title: str = Field(description="כותרת קצרה")
    description: str = Field(description="תיאור המידע")
    source_file: str = Field(description="שם הקובץ ממנו חולץ המידע")

class ProjectData(BaseModel):
    """אוסף כל הנתונים המובנים מהפרויקט"""
    items: List[ProjectItem]

# --- תהליך ה-Extraction ---

def extract_structured_data(documents):
    print("🪄 מתחיל חילוץ נתונים מובנים (Data Extraction)...")
    llm = OpenAI(model="gpt-4o-mini")
    
    # הגדרת ה"תוכנית" לחילוץ
    program = OpenAIPydanticProgram.from_defaults(
        output_cls=ProjectData,
        llm=llm,
        prompt_template_str=(
            "עבור הטקסט הבא ממסמכי הפרויקט, חלץ את כל ההחלטות (Decisions), "
            "החוקים/הנחיות (Rules) והאזהרות (Warnings).\n"
            "טקסט:\n{input_str}"
        ),
        verbose=True
    )

    all_extracted_items = []
    for doc in documents:
        print(f"📄 סורק קובץ: {doc.metadata.get('file_name')}")
        try:
            output = program(input_str=doc.text)
            # הוספת שם הקובץ לכל פריט
            for item in output.items:
                item.source_file = doc.metadata.get('file_name', 'unknown')
                all_extracted_items.append(item.dict())
        except Exception as e:
            print(f"⚠️ שגיאה בחילוץ מקובץ: {e}")

    # שמירה לקובץ JSON (כדי שנוכל לתשאל אותו בשלב הבא)
    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_items, f, indent=4, ensure_ascii=False)
    print("✅ החילוץ הסתיים! הנתונים נשמרו ב-extracted_data.json")

# --- תהליך ה-Ingestion הרגיל ---

def run_ingestion():
    print("🚀 מתחיל תהליך Ingestion...")
    reader = SimpleDirectoryReader(
        input_dir="./my_web_api", 
        recursive=True, 
        required_exts=[".md", ".txt"], # מכריח אותו לחפש קבצי טקסט
        exclude_hidden=False           # מאפשר לו להיכנס גם לתיקיות עם נקודה כמו .cursor
    )    
    documents = reader.load_data()

    # שלב ג': חילוץ נתונים מובנים
    extract_structured_data(documents)

    # שלב א'+ב': יצירת אינדקס וקטורי (כמו קודם)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="./storage")
    print("✅ האינדקס הוקטורי נוצר ונשמר בתיקיית storage")

if __name__ == "__main__":
    run_ingestion()