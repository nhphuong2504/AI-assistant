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

# Simple, clean CSS
st.markdown("""
<style>
    /* Hide only footer, keep header for sidebar toggle */
    footer {visibility: hidden;}
    
    /* Ensure sidebar is visible and accessible */
    [data-testid="stSidebar"] {
        visibility: visible !important;
    }
    
    /* Make sidebar toggle button more visible */
    [data-testid="stHeader"] button[kind="header"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
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

# Title
st.title("💬 Online Retail Data Assistant")
st.caption("Ask me anything about your retail data")


# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome. I’m your Retail Data Assistant, here to help with data analysis and insights.\n\n"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show typing indicator if processing
if st.session_state.get('is_typing', False):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.write("")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Set typing state
    st.session_state.is_typing = True
    
    # Display assistant response placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Make API request
            response = requests.post(
                f"{API_URL}/ask-langchain",
                json={
                    "question": prompt,
                    "use_memory": use_memory,
                    "thread_id": "default"
                },
                timeout=300
            )
            
            if response.status_code == 200:
                payload = response.json()
                answer = payload["answer"]
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"❌ Error: {response.text}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except requests.exceptions.Timeout:
            error_msg = "⏱️ Request timed out. The query may be too complex. Please try again or simplify your question."
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear typing state
    st.session_state.is_typing = False
