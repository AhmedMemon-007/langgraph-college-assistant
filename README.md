# 🎓 Intelligent College Assistant

An AI-powered **College Assistant** built using **LangGraph**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Mistral AI**, and **Streamlit**.

The assistant intelligently classifies student queries into different categories and routes them to the appropriate knowledge source before generating an accurate, context-aware response.

---

## 🚀 Features

- 🤖 AI-powered conversational assistant
- 📚 Retrieval-Augmented Generation (RAG)
- 🧠 LLM-based query classification
- 🔀 Conditional routing using LangGraph
- 📘 Academic Handbook retrieval
- 💳 Fee Structure retrieval
- 💬 General conversation support
- 🎓 Personalized responses based on the student's programme
- 🌐 Interactive Streamlit interface
- ⚡ Fast semantic search using FAISS

---

## 🏗️ System Architecture

```text
                     Student Question
                            │
                            ▼
                  LLM Query Classifier
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
 Academic Retriever    Fee Retriever      General Chat
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                 Response Generation
                            │
                            ▼
                 Personalized Response
```

---

## 🧠 Workflow

1. Student selects their programme.
2. Student enters a question.
3. The LLM classifies the query into one of three categories:
   - Academic
   - Fee
   - General
4. Depending on the category:
   - **Academic** → Retrieves relevant information from the Academic Handbook.
   - **Fee** → Retrieves relevant information from the Fee Structure.
   - **General** → Uses the LLM directly without document retrieval.
5. The assistant generates a personalized response based on the student's selected programme.

---

## 📚 Knowledge Sources

The assistant currently retrieves information from two official college documents:

- 📘 Academic Handbook
- 💳 Fee Structure

Both documents are:

- Loaded using **PyPDFLoader**
- Split using **Recursive Character Text Splitter**
- Embedded using **HuggingFace Embeddings**
- Indexed using **FAISS**

---

## ⚙️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Interface |
| LangGraph | Workflow Orchestration |
| LangChain | LLM Framework |
| FAISS | Vector Store |
| HuggingFace Embeddings | Semantic Embeddings |
| Mistral AI | Large Language Model |
| PyPDFLoader | PDF Processing |
| Recursive Character Text Splitter | Document Chunking |
| python-dotenv | Environment Variables |

---

## 📁 Project Structure

```text
Intelligent_College_Assistant/
│
├── app.py                     # Streamlit user interface
├── main.py                    # Core LangGraph workflow
├── academics_handbook.pdf     # Academic knowledge base
├── fee_structure.pdf          # Fee knowledge base
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚡ Installation

### Clone the repository

```bash
git clone https://github.com/AhmedMemon-007/langgraph-college-assistant.git
```

### Navigate to the project

```bash
cd langgraph-college-assistant
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
MISTRAL_API_KEY=your_api_key_here
```

### Run the application

```bash
streamlit run app.py
```

---

## 💬 Example Questions

### Academic

- What is the minimum attendance requirement?
- What are the graduation requirements?
- What is the grading policy?
- Can I appear in supplementary exams?

### Fee

- What is the tuition fee?
- Are scholarships available?
- What is the refund policy?
- What are the late payment charges?

### General

- Hello
- Who are you?
- Thank you
- What can you do?

---

## 🔀 Query Routing

| Query Type | Knowledge Source |
|------------|------------------|
| Attendance | Academic Handbook |
| Examinations | Academic Handbook |
| Credits | Academic Handbook |
| Degree Requirements | Academic Handbook |
| Tuition | Fee Structure |
| Scholarships | Fee Structure |
| Refund Policy | Fee Structure |
| Greetings | General LLM |

---

## 🌟 Key Highlights

- Multi-source Retrieval-Augmented Generation (RAG)
- Intelligent LLM-based query routing
- State-based workflow using LangGraph
- Personalized responses
- Modular architecture
- Easily extensible to additional document sources

---

## 🚀 Future Improvements

- Conversation memory
- Multi-agent workflow
- Hybrid Search (BM25 + FAISS)
- Citation of retrieved document sources
- Student login system
- Voice assistant
- Support for multiple universities
- Admin dashboard
- Course recommendation system
- Timetable assistant

---

## 👨‍💻 Author

**Muhammad Ahmed Memon**

- **GitHub:** https://github.com/AhmedMemon-007
- **LinkedIn:** https://www.linkedin.com/in/immohammadahmed/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.
