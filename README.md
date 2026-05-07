# 🚀 Hybrid RAG Agent - Project Management Assistant

סוכן AI מתקדם המבוסס על ארכיטקטורת **Event-Driven RAG** ו-**Structured Data Extraction**. 
המערכת מסוגלת לענות על שאלות מורכבות מתוך מסמכי פרויקט (Markdown) תוך שילוב בין חיפוש סמנטי לשליפה מובנית.

## 🌟 יכולות המערכת
- **שלב א' (MVP):** חיפוש סמנטי מבוסס Embeddings ו-Vector Storage.
- **שלב ב' (Architecture):** ניהול זרימה מבוסס אירועים (LlamaIndex Workflows) עם ולידציות למניעת הזיות.
- **שלב ג' (Extraction):** חילוץ אוטונומי של החלטות, חוקים ואזהרות לקובץ JSON וניתוב חכם (Routing) לפי סוג השאלה.

## 🛠 טכנולוגיות
- **Framework:** LlamaIndex (Workflows, PydanticProgram)
- **LLM:** OpenAI `gpt-4o-mini`
- **Frontend:** Gradio
- **Data Handling:** Pydantic, JSON, SimpleDirectoryReader

## 🚀 איך להריץ?
1. **התקנת ספריות:**
   ```bash
   uv pip install llama-index llama-index-llms-openai llama-index-program-openai gradio python-dotenv pydantic
   הגדרת משתני סביבה:
צרי קובץ .env והוסיפי את ה-API Key שלך:
OPENAI_API_KEY=your_key_here

Inception & Extraction (הכנת הנתונים):

Bash
python ingest.py
הפעלת הסוכן:

Bash
python app.py
הסוכן יהיה זמין בדפדפן בכתובת: http://127.0.0.1:7860

דוגמאות לשאלות שהסוכן יודע לענות
שאלה מובנית: "תן לי רשימה של כל ההחלטות והחוקים בפרויקט."

שאלה סמנטית: "מה המגבלה של ספריית הגרפים ואיך מתמודדים איתה?"

שאלה טכנית: "מהן דרישות הזיכרון עבור Docker?"
הפרויקט כולל תרשים זרימה אינטראקטיבי שנוצר אוטומטית באמצעות draw_all_possible_flows