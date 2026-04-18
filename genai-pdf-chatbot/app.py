import os
import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# ================================
# 🔑 API KEY INPUT
# ================================
st.set_page_config(page_title="GenAI PDF Chatbot", layout="wide")

st.title("📄🤖 GenAI PDF Chatbot (Gemini)")

api_key = st.text_input("Enter Google API Key", type="password")

# Only proceed if API key is provided
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

    # ================================
    # 🔍 AUTO-DETECT MODELS
    # ================================
    @st.cache_resource
    def get_models():
        embed_model = None
        chat_model = None

        for m in genai.list_models():
            methods = m.supported_generation_methods

            if "embedContent" in methods and not embed_model:
                embed_model = m.name

            if "generateContent" in methods and not chat_model:
                chat_model = m.name

        return embed_model, chat_model

    EMBED_MODEL, CHAT_MODEL = get_models()

    # ================================
    # 📄 PDF PROCESSING
    # ================================
    def load_pdf(uploaded_file):
        pdf = PdfReader(uploaded_file)
        text = ""

        for page in pdf.pages:
            text += page.extract_text() or ""

        return text

    def split_text(text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_text(text)

    @st.cache_resource
    def create_vector_store(text_chunks, embed_model_name):
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_model_name)
        return FAISS.from_texts(text_chunks, embedding=embeddings)

    # ================================
    # 🤖 QA FUNCTION
    # ================================
    def ask_question(vector_store, query, chat_model_name):
        docs = vector_store.similarity_search(query)

        llm = ChatGoogleGenerativeAI(
            model=chat_model_name,
            temperature=0.3
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.invoke({
            "input_documents": docs,
            "question": query
        })

        return response["output_text"]

    # ================================
    # 📂 FILE UPLOAD
    # ================================
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text = load_pdf(uploaded_file)
            chunks = split_text(text)
            vector_store = create_vector_store(chunks, EMBED_MODEL)

        st.success("✅ PDF processed successfully!")

        # ================================
        # 💬 CHAT UI
        # ================================
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.chat_input("Ask something about the PDF...")

        if query:
            answer = ask_question(vector_store, query, CHAT_MODEL)

            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))

        # Display chat
        for role, msg in st.session_state.chat_history:
            if role == "You":
                st.chat_message("user").write(msg)
            else:
                st.chat_message("assistant").write(msg)
else:
    st.warning("Please enter your Google API Key to proceed.")