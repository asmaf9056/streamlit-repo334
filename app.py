import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import time
from typing import List

# Page configuration
st.set_page_config(
    page_title="Datacrumbs Website Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Chat with Datacrumbs Website")
st.markdown("Ask questions about the Datacrumbs website content or any creative questions!")

# Sidebar for API key input
with st.sidebar:
    st.header("🔐 Configuration")
    api_key = st.text_input(
        "Enter your Google Gemini API Key:",
        type="password",
        help="Your API key will not be stored and is only used for this session."
    )
    
    if api_key:
        st.success("API Key provided ✅")
    else:
        st.warning("Please provide your Google Gemini API key to continue.")
    
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    **Website-related:**
    - What is Datacrumbs about?
    - What services does Datacrumbs offer?
    - Who is the team behind Datacrumbs?
    
    **Creative:**
    - Write a haiku about data science
    - What's the meaning of life?
    """)

# Initialize session state
if 'website_loaded' not in st.session_state:
    st.session_state.website_loaded = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_data
def load_website_content():
    """Load and process website content with caching"""
    try:
        with st.spinner("Loading Datacrumbs website content..."):
            # Load website content
            loader = WebBaseLoader("https://datacrumbs.org/")
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            
            # Split documents
            split_docs = text_splitter.split_documents(documents)
            
            return split_docs
    except Exception as e:
        st.error(f"Error loading website: {str(e)}")
        return []

def get_answer(question: str, documents: List[Document], api_key: str):
    """Get answer using LangChain and Google Gemini"""
    try:
        # Initialize the Google Gemini LLM
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        
        # Get answer
        with st.spinner("Thinking..."):
            # Add rate limiting
            time.sleep(1)
            
            if documents:
                # If we have website documents, use them for context
                response = chain.run(input_documents=documents, question=question)
            else:
                # For creative questions, create a simple document
                creative_doc = Document(
                    page_content="This is a creative question that doesn't require website context.",
                    metadata={"source": "creative"}
                )
                response = chain.run(input_documents=[creative_doc], question=question)
            
            return response
            
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            return "⚠️ API quota exceeded. Please wait and try again later, or check your Google Cloud Console for quota limits."
        else:
            return f"Error: {str(e)}"

# Main application logic
if api_key:
    # Load website content if not already loaded
    if not st.session_state.website_loaded:
        documents = load_website_content()
        if documents:
            st.session_state.documents = documents
            st.session_state.website_loaded = True
            st.success(f"✅ Website loaded successfully! Found {len(documents)} content chunks.")
        else:
            st.error("Failed to load website content.")
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Question {i+1}: {q[:50]}..."):
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {a}")
    
    # Question input
    st.subheader("❓ Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is Datacrumbs about?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Quick question buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("What is Datacrumbs?"):
            question = "What is Datacrumbs about and what do they do?"
            ask_button = True
    
    with col2:
        if st.button("Services offered"):
            question = "What services or products does Datacrumbs offer?"
            ask_button = True
    
    with col3:
        if st.button("Team info"):
            question = "Who is behind Datacrumbs and what's their background?"
            ask_button = True
    
    with col4:
        if st.button("Creative question"):
            question = "Write a short poem about the beauty of data analysis and insights."
            ask_button = True
    
    # Process question
    if ask_button and question:
        # Determine if question is about the website or creative
        website_keywords = ['datacrumbs', 'website', 'service', 'team', 'about', 'offer', 'company']
        is_website_question = any(keyword in question.lower() for keyword in website_keywords)
        
        # Get answer
        if is_website_question and st.session_state.documents:
            answer = get_answer(question, st.session_state.documents, api_key)
        else:
            # For creative questions, use empty documents
            answer = get_answer(question, [], api_key)
        
        # Display answer
        st.subheader("🤖 Answer")
        st.write(answer)
        
        # Add to chat history
        st.session_state.chat_history.append((question, answer))
        
        # Clear input
        st.session_state.question_input = ""
        
        # Rerun to update the interface
        st.experimental_rerun()

else:
    st.info("👈 Please enter your Google Gemini API key in the sidebar to start chatting!")
    
    # Instructions for getting API key
    with st.expander("🔑 How to get a Google Gemini API Key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy the generated API key
        5. Paste it in the sidebar
        
        **Important:** Keep your API key secure and never share it publicly!
        """)

# Footer
st.markdown("---")
st.markdown("**⚠️ Security Note:** This app uses secure practices - your API key is not stored and is only used during your session.")

# Usage instructions
with st.expander("📖 How to use this app"):
    st.markdown("""
    1. **Enter API Key:** Provide your Google Gemini API key in the sidebar
    2. **Wait for Loading:** The app will automatically load the Datacrumbs website content
    3. **Ask Questions:** Type your question or use the quick question buttons
    4. **View Answers:** The AI will respond based on website content or general knowledge
    
    **Question Types:**
    - **Website-related:** Questions about Datacrumbs content will use the loaded website data
    - **Creative/General:** Other questions will be answered using the AI's general knowledge
    
    **Tips:**
    - Be specific in your questions for better answers
    - Check the chat history to see previous Q&As
    - If you hit rate limits, wait a few minutes before asking more questions
    """)
