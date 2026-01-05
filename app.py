import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIG & THEME ---
st.set_page_config(page_title="UET POLICY ADVISOR", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #001220 0%, #000000 100%); color: #ffffff; }
    [data-testid="stChatMessage"] p { color: #ffffff !important; }
    [data-testid="stChatMessage"] h1, h2, h3 { color: #00f2fe !important; }
    .uet-title {
        font-size: 60px; font-weight: 900; text-align: center;
        background: linear-gradient(180deg, #ffffff 0%, #4facfe 40%, #00f2fe 60%, #005a92 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        filter: drop-shadow(0px 10px 10px rgba(0,0,0,0.8));
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    # Yeh 3 linein har kism ka rasta check karengi
    possible_paths = ["UET_Rules.pdf", "./UET_Rules.pdf"]
    pdf_path = None
    
    for p in possible_paths:
        if os.path.exists(p):
            pdf_path = p
            break
            
    if pdf_path:
        try:
            loader = PyPDFLoader(pdf_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            st.error(f"PDF Reading Error: {e}")
            return None
    else:
        # Yeh line aapko bataye gi ke Streamlit ko kon konsi files nazar aa rahi hain
        files_in_dir = os.listdir('.')
        st.error(f"File not found! Available files: {files_in_dir}")
        return Nones

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00f2fe;'>⚙️ SYSTEM OS</h2>", unsafe_allow_html=True)
    st.info("🤖 CORE: GEMINI-3-FLASH")
    st.success("🔒 SECURE: ENCRYPTED")

# --- 4. MAIN INTERFACE ---
st.markdown('<p class="uet-title">UET POLICY ADVISOR</p>', unsafe_allow_html=True)

if vector_db:
    GEMINI_API_KEY = "AIzaSyBUWOJgvmABGxUJKuDhagM3iDPBbvx5kkA"
    genai.configure(api_key=GEMINI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about UET rules..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Scanning rules..."):
                docs = vector_db.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])
                try:
                    model = genai.GenerativeModel('models/gemini-3-flash-preview')
                    response = model.generate_content(f"Context: {context}\nQuestion: {prompt}")
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.error("UET_Rules.pdf NOT FOUND! Check if file is on GitHub root.")

