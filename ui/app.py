import streamlit as st
import requests
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Retail Data Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THEME / CSS ---
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
.h-title { font-size: 22px; font-weight: 700; line-height: 1.1; margin:0; color: #000000; }
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
.bot svg { width: 18px; height: 18px; opacity: 1; }
.bot svg path, .bot svg rect { stroke: #4b5563; }

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

/* --- FIXED BOTTOM BAR STYLING --- */
/* This creates the white background bar fixed at bottom */
.input-wrap {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  background: #f6f7f9;
  border-top: 1px solid #e9ecef;
  padding: 20px 0; /* Increased padding */
  z-index: 999;
  height: 80px; /* Fixed height to cover bottom area */
  pointer-events: none; /* Let clicks pass through to the widgets */
}

/* --- WIDGET STYLING --- */
/* --- WIDGET STYLING --- */

/* 1. Reset the outermost container (Streamlit wrapper) */
/* This is often where the 'black corners' hide */
[data-testid="stTextInput"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* 2. Reset the BaseWeb container (The immediate parent of the input) */
div[data-baseweb="input"] {
    background-color: transparent !important;
    border: none !important; 
    border-radius: 24px !important; /* Match your desired radius */
    box-shadow: none !important;
}

/* 3. Style the Actual Input Field */
/* We apply the border and background HERE only */
.stTextInput input {
    height: 48px;
    border-radius: 24px !important;
    border: 1px solid #e5e7eb !important; /* Light gray border */
    background: #ffffff !important;        /* Pure white background */
    padding-left: 20px !important;
    color: #000000 !important;
    caret-color: #000000 !important;
}

/* 4. Handle Focus State */
/* When clicking inside, change the border color of the input itself */
.stTextInput input:focus {
    border: 1px solid #0ea5a4 !important; /* Teal focus border */
    outline: none !important;
}

/* 5. Placeholder Styling */
.stTextInput input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}



/* Force standard columns to align content vertically */
[data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Target the Send Button specifically in the layout */
/* We assume it's the button in the narrow column */
[data-testid="column"] .stButton button,
[data-testid="column"] .stButton > button,
button[data-testid="baseButton-secondary"],
.stButton button[kind="secondary"],
div[data-testid="column"]:nth-child(2) button {
    width: 48px !important;
    height: 48px !important;
    border-radius: 50% !important;
    background-color: #0ea5a4 !important;
    background: #0ea5a4 !important;
    color: white !important;
    border: none !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
}
[data-testid="column"] .stButton button:hover,
[data-testid="column"] .stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover,
.stButton button[kind="secondary"]:hover,
div[data-testid="column"]:nth-child(2) button:hover {
    background-color: #0c8d8c !important;
    background: #0c8d8c !important;
    color: white !important;
}
[data-testid="column"] .stButton button:active,
[data-testid="column"] .stButton > button:active,
button[data-testid="baseButton-secondary"]:active,
.stButton button[kind="secondary"]:active,
div[data-testid="column"]:nth-child(2) button:active {
    transform: translateY(1px);
    background-color: #0ea5a4 !important;
    background: #0ea5a4 !important;
    color: white !important;
}
[data-testid="column"] .stButton button:focus,
[data-testid="column"] .stButton > button:focus,
button[data-testid="baseButton-secondary"]:focus,
.stButton button[kind="secondary"]:focus,
div[data-testid="column"]:nth-child(2) button:focus {
    background-color: #0ea5a4 !important;
    background: #0ea5a4 !important;
    color: white !important;
}

/* Hide the default Streamlit labels and spacing */
.stTextInput {
    margin-bottom: 0 !important;
}
.stButton {
    margin-bottom: 0 !important;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 0;
}

/* Thinking animation */
.thinking-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.thinking-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #6b7280;
    animation: thinking-pulse 1.4s ease-in-out infinite;
}

.thinking-dot:nth-child(1) {
    animation-delay: 0s;
}

.thinking-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.thinking-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes thinking-pulse {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: scale(0.8);
    }
    30% {
        opacity: 1;
        transform: scale(1);
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# --- 3. SETTINGS & SIDEBAR ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.title("⚙️ Settings")
    api_url_input = st.text_input("API URL", value=API_URL, help="Backend API endpoint")
    if api_url_input:
        API_URL = api_url_input
    
    st.divider()
    
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
    with st.expander("ℹ️ About"):
        st.markdown("**Retail Data Assistant**\n\nPowered by LangChain AI agent.")

# --- 4. HEADER ---
st.markdown(
    """
<div class="chat-header">
  <div class="logo">AI</div>
  <div>
    <p class="h-title"> Online-Retail Data Assistant</p>
    <p class="h-sub">Always here to help with your retail data</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- 5. STATE MANAGEMENT ---
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome. I'm your Retail Data Assistant, here to help with data analysis and insights."
        }
    ]

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

# --- 6. CALLBACK FUNCTION (CORE LOGIC) ---
def handle_input():
    """Handles user input from both 'Enter' key and Button click."""
    widget_key = f"prompt_text_{st.session_state.input_counter}"
    user_input = st.session_state.get(widget_key, "").strip()
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.is_typing = True
        st.session_state.input_counter += 1

# --- 7. RENDER CHAT HISTORY ---
BOT_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="#4b5563" stroke-width="1.8"><path d="M12 2v2"/><path d="M8 4h8"/><rect x="6" y="7" width="12" height="12" rx="3"/><path d="M9 12h.01M15 12h.01"/><path d="M9 16h6"/></svg>"""
USER_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8"><path d="M20 21a8 8 0 10-16 0"/><circle cx="12" cy="8" r="3"/></svg>"""

st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for m in st.session_state.messages:
    role = m["role"]
    content = m["content"] or ""
    text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

    if role == "assistant":
        st.markdown(f"""<div class="row assistant"><div class="bot">{BOT_SVG}</div><div class="bubble assistant">{text}</div></div>""", unsafe_allow_html=True)
    else:
        first_letter = (content.strip()[:1] or "U").upper()
        st.markdown(f"""<div class="row user"><div class="bubble user">{text}</div><div class="userIcon">{USER_SVG}</div></div>""", unsafe_allow_html=True)

if st.session_state.get('is_typing', False):
    thinking_html = """<div class="row assistant"><div class="bot">{}</div><div class="bubble assistant"><span class="thinking-dots"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span></span></div></div>""".format(BOT_SVG)
    st.markdown(thinking_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 8. FIXED INPUT AREA (THE FIX) ---
# We inject the background bar visually
st.markdown('<div class="input-wrap"></div>', unsafe_allow_html=True)

# We use standard columns for the layout. 
# vertical_alignment="center" is the KEY to fixing the offset.
c1, c2 = st.columns([12, 1], gap="small", vertical_alignment="center")

with c1:
    st.text_input(
        "",
        placeholder="Type a message...",
        label_visibility="collapsed",
        key=f"prompt_text_{st.session_state.input_counter}",
        on_change=handle_input
    )

with c2:
    st.button(
        "➤", 
        use_container_width=True, 
        key=f"send_btn_{st.session_state.input_counter}",
        on_click=handle_input
    )

# --- 9. HANDLE API RESPONSE ---
if st.session_state.get('is_typing', False) and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    try:
        response = requests.post(
            f"{API_URL}/ask-langchain",
            json={"question": last_user_msg, "use_memory": use_memory, "thread_id": "default"},
            timeout=300
        )
        if response.status_code == 200:
            payload = response.json()
            answer = payload.get("answer", "No answer provided.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {response.text}"})
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
    
    st.session_state.is_typing = False
    st.rerun()
