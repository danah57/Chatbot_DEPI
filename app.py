#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KONA UNIVERSITY CHATBOT - Premium Dark Theme
Beautiful UI inspired by modern AI assistants
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add notebooks directory to path
sys.path.append(str(Path(__file__).parent / "notebooks"))

# Import RAG system
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_system",
        Path(__file__).parent / "notebooks" / "05_rag_system.py"
    )
    rag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_module)
    RAGChatbotWithGoogle = rag_module.RAGChatbotWithGoogle
except Exception as e:
    st.error(f"❌ Could not load RAG system: {e}")
    RAGChatbotWithGoogle = None

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Study Abroad Helper",
    page_icon="utilities/icons/school.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "system_loaded" not in st.session_state:
        st.session_state.system_loaded = False

initialize_session_state()

DATA_FILE = "./data/processed/universities_data.csv"
EMBEDDINGS_FILE = "./data/processed/embeddings.pkl"
FAISS_INDEX_FILE = "./data/processed/faiss_index.bin"


@st.cache_resource
def load_rag_system():
    """Load RAG system"""
    try:
        required = {
            "Data": DATA_FILE,
            "Embeddings": EMBEDDINGS_FILE,
            "FAISS Index": FAISS_INDEX_FILE
        }
        
        missing = [f"{n}: {p}" for n, p in required.items() if not Path(p).exists()]
        
        if missing:
            st.error("❌ Missing files:")
            for f in missing:
                st.error(f"  • {f}")
            return None
        
        with st.spinner("Loading AI System..."):
            rag = RAGChatbotWithGoogle(
                data_path=DATA_FILE,
                embeddings_path=EMBEDDINGS_FILE,
                index_path=FAISS_INDEX_FILE
            )
        
        return rag
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f35 50%, #0f1419 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #93c5fd;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0f1419 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #60a5fa !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* User Message */
    [data-testid="stChatMessageContent"] {
        color: #e2e8f0;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input Field */
    .stChatInputContainer {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 15px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #60a5fa;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        color: #60a5fa !important;
        font-weight: 600;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Slider */
    .stSlider [data-testid="stTickBar"] {
        background: rgba(59, 130, 246, 0.2);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Divider */
    hr {
        border-color: rgba(59, 130, 246, 0.2);
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(59, 130, 246, 0.2) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .feature-list {
        color: #cbd5e1;
        font-size: 1.1rem;
        line-height: 2;
    }
    
    /* AI Avatar Container */
    .ai-avatar {
        text-align: center;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .avatar-circle {
        width: 200px;
        height: 200px;
        margin: 0 auto 1rem;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, rgba(30, 58, 138, 0.1) 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 5rem;
        box-shadow: 0 0 50px rgba(59, 130, 246, 0.5);
        border: 2px solid rgba(59, 130, 246, 0.3);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 50px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 80px rgba(59, 130, 246, 0.8); }
    }
    
    .avatar-text {
        color: #94a3b8;
        font-size: 1.2rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# Load custom icon
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Convert image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

icon_path = Path("utilities/icons/school.png")
icon_base64 = get_base64_image(icon_path)

if icon_base64:
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
            <img src="data:image/png;base64,{icon_base64}" style="width: 60px; height: 60px; filter: drop-shadow(0 4px 10px rgba(59, 130, 246, 0.5));">
            <div class="main-title">Study Abroad Helper</div>
        </div>
        <div class="main-subtitle">Your AI guide to studying abroad - Universities, visas, scholarships & applications</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback if icon not found
    st.markdown("""
    <div class="main-header">
        <div class="main-title"> Study Abroad Helper</div>
        <div class="main-subtitle">Your AI guide to studying abroad - Universities, visas, scholarships & applications</div>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    # System status
    if not st.session_state.system_loaded:
        if st.button(" Activate AI Assistant", use_container_width=True):
            rag = load_rag_system()
            if rag:
                st.session_state.rag_system = rag
                st.session_state.system_loaded = True
                st.success("AI Online!")
                st.rerun()
    else:
        st.success("AI Assistant Active")
        with st.expander("System Stats"):
            try:
                st.metric("Total Programs", len(st.session_state.rag_system.data))
                st.metric("Vector Index", f"{st.session_state.rag_system.index.ntotal:,}")
                st.metric("AI Model", "MiniLM-L6")
                llm = "Gemini 2.0 ⚡" if st.session_state.rag_system.llm else "Basic Mode"
                st.metric("Language Model", llm)
            except:
                pass
    
    st.divider()
    
    # Settings
    st.markdown("### Search Settings")
    k = st.slider("Results to show", 3, 10, 5, help="Number of universities to retrieve")
    
    st.divider()
    
    # Examples
    st.markdown("### Try These")
    examples = [
        "🔍 Cheap engineering programs",
        "💼 Best MBA programs",
        "⚖️ Compare CS masters",
        "📝 Low IELTS requirements",
        "💰 Under $10k universities"
    ]
    
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    st.caption(" Powered by Gemini AI")



if not st.session_state.system_loaded:
    # Welcome Screen with Avatar
    st.markdown("""
    <div class="ai-avatar">
        <div class="avatar-circle">🤖</div>
        <div class="avatar-text">
            Your guide to studying abroad.<br>
            Ask me about universities, visas, scholarships, and applications.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">🎯 How I Can Help</div>
            <div class="feature-list">
                🔍 Search universities & programs<br>
                💰 Compare fees & costs<br>
                📚 Get detailed program info<br>
                🌍 Find programs by language<br>
                ⏱️ Check duration & requirements<br>
                🎓 Get personalized recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("💬 Ask me anything about studying abroad..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Analyzing your query..."):
                    
                    # Call RAG system
                    result = st.session_state.rag_system.answer(user_input, k=k)
                    
                    if result['count'] == 0:
                        response = "Sorry, I couldn't find any matching programs. Try rephrasing your query or being more specific."
                        st.warning(response)
                    else:
                        # Display response
                        response = result['response']
                        st.markdown(response)
                        
                        # Show detailed results
                        with st.expander(f" View {result['count']} Detailed Results"):
                            programs = result['programs']
                            
                            for i, (idx, row) in enumerate(programs.iterrows(), 1):
                                st.markdown(f"### {i}. {row.get('program', 'N/A')}")
                                st.markdown(f"**🏛️ University:** {row.get('university_name', 'N/A')}")
                                
                                cols = st.columns(4)
                                
                                # Fees
                                fees = row.get('fees', 0)
                                try:
                                    f = float(fees) if fees else 0
                                    cols[0].metric("💰 Fees", f"${f:,.0f}" if f > 0 else "N/A")
                                except:
                                    cols[0].metric("💰 Fees", "N/A")
                                
                                # Duration
                                cols[1].metric("⏱️ Duration", str(row.get('duration', 'N/A')))
                                
                                # IELTS
                                ielts = row.get('ielts', 0)
                                try:
                                    ie = float(ielts) if ielts else 0
                                    cols[2].metric("📝 IELTS", f"{ie}" if ie > 0 else "N/A")
                                except:
                                    cols[2].metric("📝 IELTS", "N/A")
                                
                                # TOEFL
                                toefl = row.get('toefl', 0)
                                try:
                                    tf = float(toefl) if toefl else 0
                                    cols[3].metric("📝 TOEFL", f"{tf}" if tf > 0 else "N/A")
                                except:
                                    cols[3].metric("📝 TOEFL", "N/A")
                               
                                
                                if i < len(programs):
                                    st.divider()
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                err = f"❌ Oops! Something went wrong: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


st.divider()
