import os
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CUSTOM UI STYLING
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .main-header {
            padding: 1.7rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #14213d, #1d4e89);
            color: white;
            margin-bottom: 1.5rem;
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.3rem;
        }

        .main-header p {
            margin-top: 0.5rem;
            margin-bottom: 0;
            opacity: 0.9;
        }

        .sidebar-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 0.8rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LANGGRAPH STATE
# =========================================================

class State(TypedDict, total=False):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# =========================================================
# 4. EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# 5. BUILD PDF RETRIEVER
# =========================================================

@st.cache_resource(show_spinner=False)
def build_retriever(pdf_path: str):
    file_path = Path(pdf_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"The file '{pdf_path}' was not found."
        )

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=load_embedding_model(),
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


# =========================================================
# 6. LOAD LLM
# =========================================================

@st.cache_resource(show_spinner=False)
def load_llm():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if not mistral_api_key:
        raise ValueError(
            "MISTRAL_API_KEY is missing from the .env file."
        )

    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.4,
        api_key=mistral_api_key,
    )


# =========================================================
# 7. BUILD LANGGRAPH WORKFLOW
# =========================================================

@st.cache_resource(show_spinner=False)
def build_graph():
    academic_retriever = build_retriever(
        "academics_handbook.pdf"
    )

    fee_retriever = build_retriever(
        "fee_structure.pdf"
    )

    llm = load_llm()

    # -----------------------------------------------------
    # Classifier Node
    # -----------------------------------------------------

    def classifier_node(state: State) -> dict:
        """
        Classifies the latest student query into:
        academic, fee, or general.
        """

        last_message = state["messages"][-1].content

        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"

            "Use 'academic' for questions about attendance, exams, grading, "
            "credits, promotion, course structure, summer training, semester "
            "rules, or degree requirements.\n"

            "Use 'fee' for questions about tuition, fee payment, refunds, "
            "late charges, scholarships, installments, or any money-related "
            "college topic.\n"

            "Use 'general' for greetings, casual conversation, or anything "
            "not related to academic rules or college fees.\n\n"

            f"Student query: {last_message}\n\n"

            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)

        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"

        elif "fee" in category:
            category = "fee"

        else:
            category = "general"

        return {
            "query_type": category
        }

    # -----------------------------------------------------
    # Academic RAG Node
    # -----------------------------------------------------

    def academic_rag_node(state: State) -> dict:
        """
        Retrieves relevant information from the
        academics handbook PDF.
        """

        query = state["messages"][-1].content

        documents = academic_retriever.invoke(query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        return {
            "retrieved_context": context
        }

    # -----------------------------------------------------
    # Fee RAG Node
    # -----------------------------------------------------

    def fee_rag_node(state: State) -> dict:
        """
        Retrieves relevant information from the
        fee structure PDF.
        """

        query = state["messages"][-1].content

        documents = fee_retriever.invoke(query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        return {
            "retrieved_context": context
        }

    # -----------------------------------------------------
    # General Node
    # -----------------------------------------------------

    def general_node(state: State) -> dict:
        """
        General conversation does not require PDF retrieval.
        """

        return {
            "retrieved_context": "NO_RETRIEVAL_NEEDED"
        }

    # -----------------------------------------------------
    # Response Node
    # -----------------------------------------------------

    def response_node(state: State) -> dict:
        """
        Generates the final answer based on the selected
        student programme and retrieved context.
        """

        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant speaking with a "
                f"{programme} student.\n\n"

                f"Student message:\n{query}\n\n"

                "Respond naturally, clearly, and briefly. "
                "Do not invent college-specific rules, fees, deadlines, "
                "or policies."
            )

        else:
            prompt = (
                f"You are a college assistant helping a {programme} student.\n\n"

                "Answer the student's question using only the official "
                "college document context provided below.\n"

                "If the answer is not available in the context, clearly say "
                "that the available document does not contain enough "
                "information.\n"

                f"If the context contains different information for different "
                f"programmes, prioritize the information relevant to "
                f"{programme}.\n\n"

                f"Official document context:\n"
                f"{context}\n\n"

                f"Student question:\n"
                f"{query}\n\n"

                "Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)

        return {
            "messages": [
                AIMessage(
                    content=response.content.strip()
                )
            ]
        }

    # -----------------------------------------------------
    # Router Function
    # -----------------------------------------------------

    def route_query(state: State) -> str:
        query_type = state["query_type"]

        if query_type == "academic":
            return "academic_rag"

        elif query_type == "fee":
            return "fee_rag"

        return "general"

    # -----------------------------------------------------
    # Build Graph
    # -----------------------------------------------------

    graph = StateGraph(State)

    graph.add_node(
        "classifier",
        classifier_node,
    )

    graph.add_node(
        "academic_rag",
        academic_rag_node,
    )

    graph.add_node(
        "fee_rag",
        fee_rag_node,
    )

    graph.add_node(
        "general",
        general_node,
    )

    graph.add_node(
        "response",
        response_node,
    )

    # -----------------------------------------------------
    # Add Graph Edges
    # -----------------------------------------------------

    graph.add_edge(
        START,
        "classifier",
    )

    graph.add_conditional_edges(
        "classifier",
        route_query,
        {
            "academic_rag": "academic_rag",
            "fee_rag": "fee_rag",
            "general": "general",
        },
    )

    graph.add_edge(
        "academic_rag",
        "response",
    )

    graph.add_edge(
        "fee_rag",
        "response",
    )

    graph.add_edge(
        "general",
        "response",
    )

    graph.add_edge(
        "response",
        END,
    )

    return graph.compile()


# =========================================================
# 8. SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your College Assistant. "
                "You can ask me questions about academic rules, "
                "attendance, examinations, fees, payments, refunds, "
                "or scholarships."
            ),
            "category": "general",
        }
    ]


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🎓 Student Profile")

    selected_programme = st.selectbox(
        "Select your programme",
        options=[
            "BCA",
            "BBA",
            "B.Com (H)",
        ],
    )

    st.divider()

    st.subheader("Knowledge Sources")

    st.markdown(
        """
        <div class="sidebar-card">
            <strong>📘 Academic Handbook</strong><br>
            Attendance, examinations, grading, credits, semesters
            and degree requirements.
        </div>

        <div class="sidebar-card">
            <strong>💳 Fee Structure</strong><br>
            Tuition fees, payments, refunds, scholarships,
            installments and late charges.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. "
                    "What would you like to know?"
                ),
                "category": "general",
            }
        ]

        st.rerun()


# =========================================================
# 10. MAIN PAGE HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Smart College Assistant</h1>
        <p>
            A LangGraph-powered multi-source RAG assistant for
            academic and fee-related student questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 11. INFORMATION METRICS
# =========================================================

column1, column2, column3 = st.columns(3)

with column1:
    st.metric(
        label="Selected Programme",
        value=selected_programme,
    )

with column2:
    st.metric(
        label="Knowledge Sources",
        value="2 PDFs",
    )

with column3:
    st.metric(
        label="Query Routes",
        value="3",
    )

st.caption(
    "Questions are automatically routed to the academic handbook, "
    "fee structure, or general conversation route."
)

st.divider()


# =========================================================
# 12. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        avatar = "🎓"
    else:
        avatar = "👤"

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):
        st.markdown(
            message["content"]
        )

        category = message.get("category")

        if (
            message["role"] == "assistant"
            and category
            and category != "general"
        ):
            st.caption(
                f"Information route: {category.title()}"
            )


# =========================================================
# 13. CHAT INPUT
# =========================================================

user_query = st.chat_input(
    "Ask about attendance, exams, grading, fees, refunds..."
)


# =========================================================
# 14. PROCESS USER QUERY
# =========================================================

if user_query:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):
        st.markdown(user_query)

    with st.chat_message(
        "assistant",
        avatar="🎓",
    ):
        try:
            with st.spinner(
                "Searching the relevant college information..."
            ):
                app = build_graph()

                result = app.invoke(
                    {
                        "programme": selected_programme,
                        "messages": [
                            HumanMessage(
                                content=user_query
                            )
                        ],
                    }
                )

                assistant_answer = (
                    result["messages"][-1].content
                )

                query_category = result.get(
                    "query_type",
                    "general",
                )

            st.markdown(
                assistant_answer
            )

            if query_category != "general":
                st.caption(
                    f"Information route: "
                    f"{query_category.title()}"
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": assistant_answer,
                    "category": query_category,
                }
            )

        except FileNotFoundError as error:
            error_message = (
                f"{error}\n\n"
                "Place `academics_handbook.pdf` and "
                "`fee_structure.pdf` in the same folder as `app.py`."
            )

            st.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "category": "general",
                }
            )

        except Exception as error:
            error_message = (
                f"Unable to process your question.\n\n"
                f"Error: {error}"
            )

            st.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "category": "general",
                }
            )