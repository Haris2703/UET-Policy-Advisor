import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIG & HIGH-END THEME ---
st.set_page_config(page_title="UET POLICY ADVISOR", layout="wide")

st.markdown("""
    <style>
    /* Dark Space Background */
    .stApp {
        background: radial-gradient(circle at center, #001220 0%, #000000 100%);
        color: #ffffff;
    }
    
    /* Force ALL Chat Text to be WHITE */
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] span {
        color: #ffffff !important;
        font-size: 16px;
        line-height: 1.6;
    }

    /* 🔥 MAZAY KA HEADING COLOR (Neon Cyan) */
    [data-testid="stChatMessage"] h1, 
    [data-testid="stChatMessage"] h2, 
    [data-testid="stChatMessage"] h3 {
        color: #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
        font-weight: bold;
        margin-top: 10px;
    }
    
    /* 3D METALLIC TITLE */
    .uet-title {
        font-size: 75px;
        font-weight: 900;
        text-align: center;
        text-transform: uppercase;
        background: linear-gradient(180deg, #ffffff 0%, #4facfe 40%, #00f2fe 60%, #005a92 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0px 10px 10px rgba(0,0,0,0.8)) 
                drop-shadow(0px 0px 30px rgba(0, 242, 254, 0.4));
        letter-spacing: 5px;
        margin-bottom: -10px;
        font-family: 'Arial Black', Gadget, sans-serif;
    }

    /* SUBTITLE */
    .subtitle {
        text-align: center;
        color: #4facfe;
        letter-spacing: 10px;
        font-size: 14px;
        text-transform: uppercase;
        margin-bottom: 50px;
        opacity: 0.8;
    }

    /* SIDEBAR TECH LOOK */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001a2c 0%, #000000 100%) !important;
        border-right: 2px solid #00f2fe;
    }

    /* RULEBOOK CARD */
    .rule-card {
        background: rgba(0, 242, 254, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px dashed #00f2fe;
        text-align: center;
        margin-top: 20px;
    }

    /* CHAT BUBBLES */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        border-radius: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2fe; font-size:26px; text-shadow: 0 0 10px #00f2fe;'>⚙️ SYSTEM OS</h1>",
                unsafe_allow_html=True)
    st.write("---")

    st.markdown("""
        <div class="rule-card">
            <p style="color:#00f2fe; font-size:18px; font-weight:bold; margin-bottom:0;">📚 KNOWLEDGE</p>
            <p style="color:#ffffff; font-size:12px;">UET_Rules.pdf</p>
            <div style="background:#00f2fe; height:2px; width:100%; margin:10px 0;"></div>
            <p style="color:#00ff00; font-size:11px;">DATABASE LINK: ACTIVE</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.info("🤖 **CORE:** GEMINI-3-FLASH")
    st.success("🔒 **SECURE:** ENCRYPTED LINK")
    st.warning("🌐 **VPN:** UET-INTERNAL-SECURE")

    if st.button("RESET SYSTEM"):
        st.session_state.messages = []
        st.rerun()

# --- 3. MAIN HEADER ---
st.markdown('<p class="uet-title">UET POLICY ADVISOR</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-Gen Academic Intelligence</p>',
            unsafe_allow_html=True)

# --- 4. ENGINE ---


@st.cache_resource
def load_engine():
    pdf_path = os.path.join(os.path.dirname(__file__), "UET_Rules.pdf")
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    return None


vector_db = load_engine()

# --- 5. CHAT LOGIC ---
if vector_db:
    GEMINI_API_KEY = "AIzaSyBUWOJgvmABGxUJKuDhagM3iDPBbvx5kkA"  # Use your key
    genai.configure(api_key=GEMINI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter your query for the Advisor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Decoding Policy..."):
                docs = vector_db.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])

                try:
                    model = genai.GenerativeModel(
                        'models/gemini-3-flash-preview')
                    # Prompt thora behter kiya taake headings use kare
                    full_query = f"Use context: {context}\n\nAnswer student query: {prompt}\n\nNote: Use proper headings and bullet points."
                    response = model.generate_content(full_query)

                    st.markdown(response.text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"Neural Error: {e}")
else:
    st.error("UET_Rules.pdf not found!")

