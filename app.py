import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
import time

# Page configuration
st.set_page_config(
    page_title="Datacrumbs Chat Bot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_api_key():
    """Get API key from secrets or user input"""
    # Try to get from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except:
        pass
    
    # Fall back to user input
    return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'website_loaded' not in st.session_state:
        st.session_state.website_loaded = False

@st.cache_data
def load_website_content():
    """Load and process website content"""
    try:
        # Load website using WebBaseLoader
        loader = WebBaseLoader("https://datacrumbs.org/")
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        
        split_docs = text_splitter.split_documents(documents)
        return split_docs
    except Exception as e:
        st.error(f"Error loading website: {str(e)}")
        return None

def setup_qa_chain(api_key):
    """Setup QA chain with Google Gemini"""
    try:
        # Initialize Gemini LLM
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create QA chain
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def get_answer(question, documents, qa_chain):
    """Get answer for user question"""
    try:
        # Check if question is creative/unrelated
        creative_keywords = ['story', 'poem', 'joke', 'creative', 'imagine', 'tell me about', 'what if']
        is_creative = any(keyword in question.lower() for keyword in creative_keywords)
        
        if is_creative:
            # For creative questions, use LLM directly
            response = qa_chain.llm.invoke(question)
            return response
        else:
            # For website questions, use documents as context
            response = qa_chain.run(input_documents=documents, question=question)
            return response
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Datacrumbs Chat Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with Datacrumbs website using AI</p>', unsafe_allow_html=True)
    
    # Get API key from secrets or user input
    api_key_from_secrets = get_api_key()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Setup")
        
        # API Key section
        st.subheader("Google Gemini API Key")
        
        if api_key_from_secrets:
            st.success("✅ API Key loaded from secrets")
            api_key = api_key_from_secrets
            st.info("Using pre-configured API key")
        else:
            st.info("💡 API key not found in secrets")
            api_key = st.text_input(
                "Enter your API key:",
                type="password",
                placeholder="AIza...",
                help="Get your API key from Google AI Studio"
            )
        
        st.markdown("---")
        
        # Load website button
        if st.button("📥 Load Website Content", type="primary"):
            if api_key:
                with st.spinner("Loading Datacrumbs website..."):
                    documents = load_website_content()
                    if documents:
                        st.session_state.documents = documents
                        qa_chain = setup_qa_chain(api_key)
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.session_state.website_loaded = True
                            st.success(f"✅ Loaded {len(documents)} text chunks!")
                        else:
                            st.error("❌ Failed to setup AI model")
                    else:
                        st.error("❌ Failed to load website")
            else:
                st.error("⚠️ Please enter API key first")
        
        # Status
        if st.session_state.website_loaded:
            st.success("🟢 Ready to chat!")
        else:
            st.warning("🟡 Setup required")
        
        st.markdown("---")
        
        # Sample questions
        st.subheader("💡 Try These Questions")
        sample_questions = [
            "What is Datacrumbs?",
            "What services do they offer?",
            "Who is the team?",
            "Tell me a creative story about data"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.selected_question = q
        
        # Stats
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📊 Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
    
    # Main chat area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Question input
        question = st.text_input(
            "💬 Ask your question:",
            placeholder="What would you like to know about Datacrumbs?",
            key="question_input"
        )
        
        # Handle selected sample question
        if hasattr(st.session_state, 'selected_question'):
            question = st.session_state.selected_question
            delattr(st.session_state, 'selected_question')
        
        # Ask button
        if st.button("🚀 Ask Question", type="primary") and question:
            if st.session_state.website_loaded and st.session_state.qa_chain:
                with st.spinner("Thinking..."):
                    answer = get_answer(question, st.session_state.documents, st.session_state.qa_chain)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "time": time.strftime("%H:%M:%S")
                    })
                    
                    st.rerun()
            else:
                st.error("⚠️ Please setup the system first!")
    
    with col2:
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## 💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You ({chat['time']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Assistant:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps:
        1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. **Enter API Key**: Paste it in the sidebar (or use secrets.toml)
        3. **Load Website**: Click "Load Website Content"
        4. **Ask Questions**: Type questions about Datacrumbs or creative questions
        
        ### Sample Questions:
        - **About Website**: "What is Datacrumbs about?"
        - **Creative**: "Tell me a story about data analysis"
        """)

if __name__ == "__main__":
    main()
