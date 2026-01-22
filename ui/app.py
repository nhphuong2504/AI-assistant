import streamlit as st
import pandas as pd
import requests
import os
import time

# Configure page
st.set_page_config(
    page_title="Retail Data Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful chatbot design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container - centered and constrained */
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 1rem;
        min-height: 100vh;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat container - centered card */
    .chat-container {
        max-width: 672px;
        width: 100%;
        height: calc(100vh - 4rem);
        max-height: 800px;
        display: flex;
        flex-direction: column;
        background: white;
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid #e5e7eb;
        overflow: hidden;
    }
    
    /* Header section */
    .chat-header {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        padding: 1.25rem 1.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-bottom: 1px solid #334155;
    }
    
    .chat-header-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #3b82f6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }
    
    .chat-header-text {
        flex: 1;
    }
    
    .chat-header-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        font-weight: 600;
        color: white;
        margin: 0;
        line-height: 1.4;
    }
    
    .chat-header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.7);
        margin: 0.25rem 0 0 0;
        line-height: 1.4;
    }
    
    /* Messages area - scrollable */
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        background: #f9fafb;
    }
    
    /* Message row */
    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        animation: fadeInUp 0.3s ease-out;
    }
    
    .message-row.user {
        flex-direction: row-reverse;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Avatar circle */
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    
    .message-avatar.user {
        background: #3b82f6;
    }
    
    .message-avatar.assistant {
        background: #10b981;
    }
    
    /* Message bubble */
    .message-bubble {
        max-width: 75%;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.6;
        word-wrap: break-word;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .message-bubble.user {
        background: #1e293b;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-bubble.assistant {
        background: white;
        color: #1e293b;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 4px;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.75rem 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #9ca3af;
        animation: typingBounce 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typingBounce {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    
    /* Input section */
    .chat-input-section {
        flex-shrink: 0;
        padding: 1rem 1.5rem;
        background: white;
        border-top: 1px solid #e5e7eb;
    }
    
    .input-wrapper {
        position: relative;
        display: flex;
        align-items: flex-end;
        gap: 0.5rem;
    }
    
    .chat-textarea {
        flex: 1;
        min-height: 44px;
        max-height: 120px;
        padding: 0.75rem 3rem 0.75rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        line-height: 1.6;
        resize: none;
        overflow-y: auto;
        background: #f9fafb;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    
    .chat-textarea:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: white;
    }
    
    .chat-textarea::placeholder {
        color: #9ca3af;
    }
    
    .send-button {
        position: absolute;
        right: 0.5rem;
        bottom: 0.5rem;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #3b82f6;
        border: none;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.2s, transform 0.1s;
        flex-shrink: 0;
    }
    
    .send-button:hover:not(:disabled) {
        background: #2563eb;
        transform: scale(1.05);
    }
    
    .send-button:disabled {
        background: #d1d5db;
        cursor: not-allowed;
        opacity: 0.5;
    }
    
    .send-button svg {
        width: 16px;
        height: 16px;
    }
    
    /* Scrollbar styling */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 3rem;
    }
    
    /* Markdown content in messages */
    .message-bubble p {
        margin: 0.5rem 0;
    }
    
    .message-bubble p:first-child {
        margin-top: 0;
    }
    
    .message-bubble p:last-child {
        margin-bottom: 0;
    }
    
    .message-bubble ul, .message-bubble ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .message-bubble code {
        background: rgba(0, 0, 0, 0.05);
        padding: 0.125rem 0.25rem;
        border-radius: 4px;
        font-size: 0.875em;
    }
    
    .message-bubble.user code {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

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
                # Clear local history too
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

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False

# Chat container HTML structure
st.markdown("""
<div class="chat-container">
    <div class="chat-header">
        <div class="chat-header-avatar">🤖</div>
        <div class="chat-header-text">
            <div class="chat-header-title">Retail Data Assistant</div>
            <div class="chat-header-subtitle">Ask me anything about your retail data</div>
        </div>
    </div>
    <div class="chat-messages" id="messages-container">
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="message-row user">
            <div class="message-avatar user">👤</div>
            <div class="message-bubble user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Escape HTML in content but allow markdown-like formatting
        import html
        escaped_content = html.escape(content)
        # Convert markdown-style formatting to HTML
        import re
        escaped_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped_content)
        escaped_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped_content)
        escaped_content = re.sub(r'`(.*?)`', r'<code>\1</code>', escaped_content)
        # Convert line breaks
        escaped_content = escaped_content.replace('\n', '<br>')
        
        st.markdown(f"""
        <div class="message-row assistant">
            <div class="message-avatar assistant">🤖</div>
            <div class="message-bubble assistant">{escaped_content}</div>
        </div>
        """, unsafe_allow_html=True)

# Show typing indicator if processing
if st.session_state.is_typing:
    st.markdown("""
    <div class="message-row assistant">
        <div class="message-avatar assistant">🤖</div>
        <div class="message-bubble assistant">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Close messages container
st.markdown("</div>", unsafe_allow_html=True)

# Input section
st.markdown("""
<div class="chat-input-section">
    <div class="input-wrapper">
""", unsafe_allow_html=True)

# Use Streamlit's text_area for auto-resize, but style it
user_input = st.text_area(
    "",
    key="chat_input",
    placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)",
    height=44,
    label_visibility="collapsed"
)

# Send button
send_button = st.button("➤", key="send_button", help="Send message")

st.markdown("</div></div></div>", unsafe_allow_html=True)

# Handle user input
if send_button and user_input and not st.session_state.is_typing:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.is_typing = True
    st.rerun()

# Process the latest user message if typing
if st.session_state.is_typing and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_question = st.session_state.messages[-1]["content"]
    
    try:
        r = requests.post(
            f"{API_URL}/ask-langchain",
            json={
                "question": user_question,
                "use_memory": use_memory,
                "thread_id": "default"
            },
            timeout=300
        )
        
        if r.status_code == 200:
            payload = r.json()
            answer = payload["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            error_msg = f"❌ Error: {r.text}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timed out. The query may be too complex. Please try again or simplify your question."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.session_state.is_typing = False
    st.rerun()

# JavaScript for auto-scroll and Enter key handling
st.markdown("""
<script>
    // Auto-scroll to bottom when new messages appear
    function scrollToBottom() {
        const container = document.getElementById('messages-container');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }
    
    // Scroll on load and when messages update
    window.addEventListener('load', scrollToBottom);
    setTimeout(scrollToBottom, 100);
    
    // Handle Enter key in textarea
    const textarea = document.querySelector('textarea[data-testid*="stTextArea"]');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const sendButton = document.querySelector('button[data-testid*="baseButton-secondary"]');
                if (sendButton && !sendButton.disabled) {
                    sendButton.click();
                }
            }
        });
    }
</script>
""", unsafe_allow_html=True)
