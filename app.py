import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIG & HIGH-END THEME (Wahi Purani CSS) ---
st.set_page_config(page_title="UET POLICY ADVISOR", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at center, #001220 0%, #000000 100%);
        color: #ffffff;
    }
    
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span {
        color: #ffffff !important;
        font-size: 16px;
    }

    [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
    
    .uet-title {
        font-size: 75px; font-weight: 900; text-align: center; text-transform: uppercase;
        background: linear-gradient(180deg, #ffffff 0%, #4facfe 40%, #00f2fe 60%, #005a92 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        filter: drop-shadow(0px 10px 10px rgba(0,0,0,0.8)) drop-shadow(0px 0px 30px rgba(0, 242, 254, 0.4));
        letter-spacing: 5px; margin-bottom: -10px; font-family: 'Arial Black', Gadget, sans-serif;
    }

    .subtitle {
        text-align: center; color: #4facfe; letter-spacing: 10px; font-size: 14px;
        text-transform: uppercase; margin-bottom: 50px; opacity: 0.8;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001a2c 0%, #000000 100%) !important;
        border-right: 2px solid #00f2fe;
    }

    .rule-card {
        background: rgba(0, 242, 254, 0.05); padding: 20px; border-radius: 15px;
        border: 1px dashed #00f2fe; text-align: center; margin-top: 20px;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        border-radius: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. ENGINE (GitHub Root Friendly) ---
@st.cache_resource
def load_engine():
    # Aap ki file GitHub baahar hai, isliye direct path:
    pdf_path = "UET_Rules.pdf"
    
    if os.path.exists(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(data)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            return None
    return None

# initialize vector_db
vector_db = load_engine()

# --- 3. SIDEBAR (Wahi Security Lines) ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2fe; font-size:26px; text-shadow: 0 0 10px #00f2fe;'>⚙️ SYSTEM OS</h1>", unsafe_allow_html=True)
    st.write("---")
    
    status_color = "#00ff00" if vector_db else "#ff0000"
    status_text = "ACTIVE" if vector_db else "MISSING"
    
    st.markdown(f"""
        <div class="rule-card">
            <p style="color:#00f2fe; font-size:18px; font-weight:bold; margin-bottom:0;">📚 KNOWLEDGE</p>
            <p style="color:#ffffff; font-size:12px;">UET_Rules.pdf</p>
            <div style="background:#00f2fe; height:2px; width:100%; margin:10px 0;"></div>
            <p style="color:{status_color}; font-size:11px;">DATABASE LINK: {status_text}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.info("🤖 **CORE:** GEMINI-3-FLASH")
    st.success("🔒 **SECURE:** ENCRYPTED LINK")
    st.warning("🌐 **VPN:** UET-INTERNAL-SECURE")
    
    if st.button("RESET SYSTEM"):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN HEADER ---
st.markdown('<p class="uet-title">UET POLICY ADVISOR</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Gen Academic Intelligence</p>', unsafe_allow_html=True)

# --- 5. CHAT LOGIC ---
if vector_db is not None:
    # 🗝️ APNI KEY DALAIN
    GEMINI_API_KEY = "YAHAN_APNI_WORKING_KEY_DALAIN"
    genai.configure(api_key=GEMINI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Accessing UET Rulebook..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Decoding Policy..."):
                try:
                    docs = vector_db.similarity_search(prompt, k=3)
                    context = "\n".join([d.page_content for d in docs])
                    model = genai.GenerativeModel('models/gemini-3-flash-preview')
                    response = model.generate_content(f"Context: {context}\n\nQuestion: {prompt}")
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Neural Error: {e}")
else:
    st.warning("⚠️ Waiting for UET_Rules.pdf. Make sure it's in the root of your GitHub repo.")
