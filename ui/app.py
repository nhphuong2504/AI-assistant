import streamlit as st
import requests
import os

# Configure page
st.set_page_config(
    page_title="Retail Data Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME / CSS (match screenshot) ---
st.markdown(
    """
<style>
/* Page background */
.stApp { background: #f6f7f9; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 18px; padding-bottom: 90px; max-width: 1200px; }

/* Ensure sidebar is visible and accessible */
[data-testid="stSidebar"] {
    visibility: visible !important;
}

/* Make sidebar toggle button more visible */
[data-testid="stHeader"] button[kind="header"] {
    background-color: #0ea5a4;
    color: white;
}

/* Header */
.chat-header {
  display:flex; align-items:center; gap:14px;
  padding: 10px 6px 14px 6px;
  border-bottom: 1px solid #e9ecef;
  margin-bottom: 14px;
}
.logo {
  width: 46px; height: 46px; border-radius: 50%;
  background: #0ea5a4; color: white;
  display:flex; align-items:center; justify-content:center;
  font-weight: 700; font-size: 18px;
}
.h-title { font-size: 22px; font-weight: 700; line-height: 1.1; margin:0; }
.h-sub   { font-size: 14px; color:#6b7280; margin:2px 0 0 0; }

/* Chat rows */
.row { display:flex; align-items:flex-start; gap:14px; margin: 16px 0; }
.row.assistant { justify-content:flex-start; }
.row.user      { justify-content:flex-end; }

/* Icons */
.bot {
  width: 34px; height: 34px; border-radius: 50%;
  background: #eef0f3; display:flex; align-items:center; justify-content:center;
  border: 1px solid #e6e8eb;
  flex: 0 0 auto;
}
.bot svg { width: 18px; height: 18px; opacity: .75; }

.userCircle {
  width: 46px; height: 46px; border-radius: 50%;
  background: #0ea5a4; color: white;
  display:flex; align-items:center; justify-content:center;
  font-weight: 700; font-size: 16px;
  flex: 0 0 auto;
}
.userIcon {
  width: 46px; height: 46px; border-radius: 50%;
  background: #0ea5a4;
  display:flex; align-items:center; justify-content:center;
  flex: 0 0 auto;
}
.userIcon svg { width: 22px; height: 22px; }

/* Bubbles */
.bubble {
  max-width: 760px;
  padding: 14px 18px;
  border-radius: 18px;
  background: #ffffff;
  border: 1px solid #eef0f3;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
  color: #111827;
}
.bubble.assistant { border-top-left-radius: 10px; }
.bubble.user {
  background: #0ea5a4; color: #ffffff;
  border-color: #0ea5a4;
  border-top-right-radius: 10px;
}

/* Bottom input bar */
.input-wrap {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  background: #f6f7f9;
  border-top: 1px solid #e9ecef;
  padding: 14px 0;
  z-index: 999;
}
.input-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.2rem;
  display: flex;
  gap: 12px;
  align-items: center;
}
.stTextInput > div > div > input {
  height: 48px;
  border-radius: 18px !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  padding-left: 14px !important;
}
.sendBtn button {
  width: 52px !important;
  height: 52px !important;
  border-radius: 50% !important;
  border: none !important;
  background: #9bd7d4 !important;
  color: #ffffff !important;
}
.sendBtn button:hover { filter: brightness(0.98); }
.sendBtn button:active { transform: translateY(1px); }

/* Make markdown blocks not add extra top/bottom spacing */
.chat-area .stMarkdown { margin: 0; }

/* Typing indicator */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Get API URL from environment or sidebar
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API URL input
    api_url_input = st.text_input("API URL", value=API_URL, help="Backend API endpoint")
    if api_url_input:
        API_URL = api_url_input
    
    st.divider()
    
    # Memory settings
    use_memory = st.checkbox(
        "💾 Use conversation memory", 
        value=True,
        help="Enable to remember context across questions"
    )
    
    if st.button("🗑️ Clear Memory", use_container_width=True):
        try:
            r = requests.post(f"{API_URL}/ask-langchain/clear-memory", timeout=10)
            if r.status_code == 200:
                st.success("Memory cleared!")
                if 'messages' in st.session_state:
                    st.session_state.messages = []
            else:
                st.error(f"Failed: {r.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Retail Data Assistant**
        
        Powered by LangChain AI agent with:
        - Multi-step reasoning
        - Conversation memory
        - Natural language understanding
        
        **Capabilities:**
        - SQL queries on transaction data
        - Customer Lifetime Value (CLV) predictions
        - Churn risk scoring
        - Churn probability predictions
        - Expected remaining lifetime
        - Customer segmentation
        """)

# --- HEADER ---
st.markdown(
    """
<div class="chat-header">
  <div class="logo">RD</div>
  <div>
    <p class="h-title">Retail Data Assistant</p>
    <p class="h-sub">Always here to help with your retail data</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome. I'm your Retail Data Assistant, here to help with data analysis and insights."
        }
    ]

# SVG Icons
BOT_SVG = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
  <path d="M12 2v2"/>
  <path d="M8 4h8"/>
  <rect x="6" y="7" width="12" height="12" rx="3"/>
  <path d="M9 12h.01M15 12h.01"/>
  <path d="M9 16h6"/>
</svg>
"""
USER_SVG = """
<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8">
  <path d="M20 21a8 8 0 10-16 0"/>
  <circle cx="12" cy="8" r="3"/>
</svg>
"""

# --- RENDER CHAT ---
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for m in st.session_state.messages:
    role = m["role"]
    # Escape HTML but preserve newlines
    content = m["content"] or ""
    text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

    if role == "assistant":
        st.markdown(
            f"""
<div class="row assistant">
  <div class="bot">{BOT_SVG}</div>
  <div class="bubble assistant">{text}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        # user: show message bubble, then circle with first letter, then profile icon
        first_letter = (content.strip()[:1] or "U").upper()
        st.markdown(
            f"""
<div class="row user">
  <div class="bubble user">{text}</div>
  <div class="userCircle">{first_letter}</div>
  <div class="userIcon">{USER_SVG}</div>
</div>
""",
            unsafe_allow_html=True,
        )

# Show typing indicator if processing
if st.session_state.get('is_typing', False):
    st.markdown(
        f"""
<div class="row assistant">
  <div class="bot">{BOT_SVG}</div>
  <div class="bubble assistant">Thinking...</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# --- FIXED BOTTOM INPUT ---
st.markdown('<div class="input-wrap"><div class="input-inner">', unsafe_allow_html=True)

# Put the widgets in columns so they sit on one line
c1, c2 = st.columns([12, 1.2], gap="small")
with c1:
    prompt = st.text_input("",
                           placeholder="Type a message...",
                           label_visibility="collapsed",
                           key="prompt_text")
with c2:
    st.markdown('<div class="sendBtn">', unsafe_allow_html=True)
    send = st.button("➤", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# --- ACTION ---
if send and prompt.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt.strip()})
    
    # Set typing state
    st.session_state.is_typing = True
    st.session_state.prompt_text = ""
    st.rerun()

# Handle API call after rerun (to show typing indicator first)
if st.session_state.get('is_typing', False) and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Get the last user message
    last_user_msg = st.session_state.messages[-1]["content"]
    
    # Make API request
    try:
        response = requests.post(
            f"{API_URL}/ask-langchain",
            json={
                "question": last_user_msg,
                "use_memory": use_memory,
                "thread_id": "default"
            },
            timeout=300
        )
        
        if response.status_code == 200:
            payload = response.json()
            answer = payload["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            error_msg = f"❌ Error: {response.text}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out. The query may be too complex. Please try again or simplify your question."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear typing state
    st.session_state.is_typing = False
    st.rerun()
