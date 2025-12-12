#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KONA UNIVERSITY CHATBOT - Premium Dark Theme
"""

# ============================================================================  
# 1️⃣ STREAMLIT PAGE CONFIG - MUST BE FIRST
# ============================================================================  
import streamlit as st
st.set_page_config(
    page_title="Study Abroad Helper",
    page_icon="utilities/icons/school.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================  
# 2️⃣ IMPORTS - after page config
# ============================================================================  
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util
import base64

# ============================================================================  
# 3️⃣ SESSION CHECK (no Streamlit commands before page config!)
# ============================================================================  
if not hasattr(st, "session_state"):
    print("ERROR: Must run with Streamlit: streamlit run app.py")
    sys.exit(1)

# ============================================================================  
# 4️⃣ ADD NOTEBOOKS PATH AND IMPORT RAG SYSTEM
# ============================================================================  
sys.path.append(str(Path(__file__).parent / "notebooks"))
try:
    spec = importlib.util.spec_from_file_location(
        "rag_system",
        Path(__file__).parent / "notebooks" / "05_rag_system.py"
    )
    rag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_module)
    RAGChatbot = rag_module.RAGChatbot
except Exception as e:
    RAGChatbot = None
    st.error(f"❌ Could not load RAG system: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================  
# 5️⃣ SESSION STATE INIT
# ============================================================================  
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "system_loaded" not in st.session_state:
    st.session_state.system_loaded = False

# ============================================================================  
# 4️⃣ FILE PATHS
# ============================================================================  
DATA_FILE = "./data/processed/universities_data.csv"
EMBEDDINGS_FILE = "./data/processed/embeddings.pkl"
FAISS_INDEX_FILE = "./data/processed/faiss_index.bin"

# ============================================================================  
# 5️⃣ LOAD RAG SYSTEM
# ============================================================================  
@st.cache_resource
def load_rag_system():
    try:
        required = {
            "Data": DATA_FILE,
            "Embeddings": EMBEDDINGS_FILE,
            "FAISS Index": FAISS_INDEX_FILE
        }

        missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
        if missing:
            st.error("❌ Missing files:\n" + "\n".join(missing))
            return None

        with st.spinner("Loading AI System..."):
            rag = RAGChatbot(
                data_path=DATA_FILE,
                embeddings_path=EMBEDDINGS_FILE,
                index_path=FAISS_INDEX_FILE
            )
        return rag
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        import traceback
        traceback.print_exc()
        return None

# Auto-load system
if not st.session_state.system_loaded:
    rag = load_rag_system()
    if rag:
        st.session_state.rag_system = rag
        st.session_state.system_loaded = True

# ============================================================================  
# 6️⃣ UI STYLE
# ============================================================================  
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f1419 0%, #1a1f35 50%, #0f1419 100%);}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.main-header {background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
              padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;}
.main-title {font-size: 2.5rem; font-weight: 800; color: white; margin: 0;}
.main-subtitle {font-size: 1.1rem; color: #93c5fd; margin-top: 0.5rem; font-weight: 300;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1f35 0%, #0f1419 100%);
                           border-right: 1px solid rgba(59, 130, 246, 0.2);}
.chat-item {color: #cbd5e1; font-size: 0.9rem; padding: 0.5rem; border-bottom: 1px solid rgba(59, 130, 246, 0.15);}
</style>
""", unsafe_allow_html=True)

# Load icon
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

icon_base64 = get_base64_image(Path("utilities/icons/school.png"))

# Header
if icon_base64:
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
            <img src="data:image/png;base64,{icon_base64}" style="width:60px;height:60px;">
            <div class="main-title">Study Abroad Helper</div>
        </div>
        <div class="main-subtitle">Your AI guide to studying abroad - Universities, visas, scholarships & applications</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Study Abroad Helper</div>
        <div class="main-subtitle">Your AI guide to studying abroad</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================  
# 7️⃣ SIDEBAR
# ============================================================================  
with st.sidebar:
    st.markdown("## 💬 Chat History")
    if not st.session_state.messages:
        st.caption("No previous messages.")
    else:
        for msg in st.session_state.messages:
            role = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"<div class='chat-item'><b>{role} {msg['role'].capitalize()}</b><br>{msg['content']}</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================================  
# 8️⃣ MAIN CHAT AREA
# ============================================================================  
if not st.session_state.system_loaded:
    st.warning("System is loading... please wait.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    if user_input := st.chat_input("💬 Ask me anything about studying abroad..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Analyzing your query..."):
                    result = st.session_state.rag_system.answer(user_input, k=5)

                    # Print full RAG result to console for debugging/inspection
                    try:
                        import json
                        # Make a shallow copy and convert DataFrame to records if present
                        result_print = dict(result)
                        programs = result_print.get('programs')
                        if hasattr(programs, 'to_dict'):
                            result_print['programs'] = programs.to_dict(orient='records')
                        print("=== RAG SYSTEM RAW RESULT ===")
                        print(json.dumps(result_print, default=str, indent=2, ensure_ascii=False))
                        print("=== END RAG RESULT ===")
                    except Exception as _err:  # fallback
                        print("RAG result (raw):", result)
                    if result['count'] == 0:
                        response = "Sorry, I couldn't find any matching programs."
                        st.warning(response)
                    else:
                        response = result['response']
                        st.markdown(response)

                        with st.expander(f" View {result['count']} Detailed Results"):
                            programs = result['programs']
                            for i, (idx, row) in enumerate(programs.iterrows(), 1):
                                st.markdown(f"### {i}. {row.get('program', 'N/A')}")
                                st.markdown(f"**🏛️ University:** {row.get('university_name', 'N/A')}")
                                cols = st.columns(4)
                                try:
                                    fees = float(row.get('fees', 0))
                                    cols[0].metric("💰 Fees", f"${fees:,.0f}" if fees else "N/A")
                                except:
                                    cols[0].metric("💰 Fees", "N/A")
                                cols[1].metric("⏱️ Duration", row.get("duration", "N/A"))
                                try:
                                    ielts = float(row.get('ielts', 0))
                                    cols[2].metric("📝 IELTS", f"{ielts}" if ielts else "N/A")
                                except:
                                    cols[2].metric("📝 IELTS", "N/A")
                                try:
                                    toefl = float(row.get('toefl', 0))
                                    cols[3].metric("📝 TOEFL", f"{toefl}" if toefl else "N/A")
                                except:
                                    cols[3].metric("📝 TOEFL", "N/A")
                                if i < len(programs):
                                    st.divider()
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

st.divider()
