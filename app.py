import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI
import datetime

# Page config
st.set_page_config(
    page_title="Datacrumbs Chat App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
    .sample-questions {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'website_content' not in st.session_state:
    st.session_state.website_content = None
if 'docs' not in st.session_state:
    st.session_state.docs = None

# Main header
st.markdown('<h1 class="main-header">🤖 Chat with Datacrumbs Website</h1>', unsafe_allow_html=True)

# Sidebar for API key and controls
with st.sidebar:
    st.header("🔑 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Enter your Google Gemini API Key:",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key configured!")
    
    st.markdown("---")
    
    # Load website button
    if st.button("🌐 Load Website Content", disabled=not api_key):
        with st.spinner("Loading Datacrumbs website content..."):
            try:
                # Load website content
                loader = WebBaseLoader("https://datacrumbs.org/")
                documents = loader.load()
                
                # Split text into chunks
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                st.session_state.docs = text_splitter.split_documents(documents)
                st.session_state.website_content = True
                
                st.success(f"✅ Loaded {len(st.session_state.docs)} document chunks!")
                
            except Exception as e:
                st.error(f"❌ Error loading website: {str(e)}")
    
    # Chat statistics
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("📊 Chat Stats")
        st.metric("Total Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not api_key:
    st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to start chatting.")
    st.info("Get your free API key from: https://makersuite.google.com/app/apikey")
else:
    # Sample questions
    st.markdown("""
    <div class="sample-questions">
        <h3>💡 Sample Questions to Try:</h3>
        <p><strong>About Datacrumbs:</strong></p>
        <ul>
            <li>What is Datacrumbs about?</li>
            <li>What services does Datacrumbs offer?</li>
            <li>Who is behind Datacrumbs?</li>
        </ul>
        <p><strong>Creative Questions:</strong></p>
        <ul>
            <li>Write a haiku about data science</li>
            <li>Tell me a story about a data scientist who discovers magic</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat messages display
    for message in st.session_state.messages:
        message_class = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{"🧑 You" if message["role"] == "user" else "🤖 Assistant"}:</strong><br>
            {message["content"]}
            <br><small>{message["timestamp"]}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # User input
    user_question = st.text_input("💬 Ask a question:", placeholder="Type your question here...")
    
    if user_question:
        # Add user message to chat
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "timestamp": timestamp
        })
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize the language model
                llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)
                
                # Check if question is about the website and we have content
                if st.session_state.website_content and any(keyword in user_question.lower() for keyword in 
                    ['datacrumbs', 'website', 'service', 'about', 'offer', 'company', 'team', 'what is']):
                    
                    # Use QA chain for website-related questions
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=st.session_state.docs, question=user_question)
                    
                else:
                    # Direct LLM response for creative or general questions
                    response = llm.invoke(user_question)
                
                # Add bot response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": timestamp
                })
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("💡 Make sure your API key is correct and you have internet connection.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit, LangChain, and Google Gemini API</p>
    <p>Website: <a href="https://datacrumbs.org/" target="_blank">datacrumbs.org</a></p>
</div>
""", unsafe_allow_html=True)
