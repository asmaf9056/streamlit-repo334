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
st.title("🛍️ Datacrumbs Sales Assistant")
st.markdown("**Your AI sales assistant for http://www.datacrumbs.org**")
st.markdown("*Ask me anything about Datacrumbs services, products, pricing, or company information!*")

# Load API key from environment variable
api_key = os.getenv('GOOGLE_API_KEY')

# Sidebar for information
with st.sidebar:
    st.header("📊 Sales Information")
    
    if api_key:
        st.success("Sales Assistant Ready ✅")
    else:
        st.error("API Key not found in environment variables")
        st.markdown("Please set GOOGLE_API_KEY environment variable")
    
    st.markdown("---")
    st.markdown("### 💼 What I can help with:")
    st.markdown("""
    - **Company Information** - About Datacrumbs
    - **Services & Products** - What we offer
    - **Pricing & Packages** - Cost information
    - **Contact Details** - How to reach us
    - **Technical Specs** - Product features
    - **Support** - Customer service info
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
        with st.spinner("🔍 Scraping Datacrumbs website content..."):
            # Load website content from the correct URL
            url = "http://www.datacrumbs.org"
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    "parse_only": None,  # Parse the entire page
                    "features": "html.parser"
                }
            )
            
            # Load the documents
            documents = loader.load()
            
            if not documents:
                st.error("No content found on the website")
                return []
            
            # Display info about loaded content
            total_chars = sum(len(doc.page_content) for doc in documents)
            st.info(f"📄 Loaded {len(documents)} page(s) with {total_chars:,} characters from {url}")
            
            # Split text into chunks for better processing
            text_splitter = CharacterTextSplitter(
                chunk_size=1500,  # Larger chunks for better context
                chunk_overlap=300,
                separator="\n\n"  # Split on paragraphs first
            )
            
            # Split documents
            split_docs = text_splitter.split_documents(documents)
            
            st.success(f"✅ Content split into {len(split_docs)} chunks for processing")
            
            return split_docs
            
    except Exception as e:
        st.error(f"❌ Error loading website: {str(e)}")
        st.info("Please check if the website is accessible and try again.")
        return []

def get_answer(question: str, documents: List[Document], api_key: str):
    """Get answer using LangChain and Google Gemini - Website content only"""
    try:
        # Check if we have website content
        if not documents or len(documents) == 0:
            return "❌ **Website Content Not Available**\n\nI need to load the Datacrumbs website content first to answer your questions. Please wait for the content to load or refresh the page."
        
        # Initialize the Google Gemini LLM with settings optimized for sales
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.2  # Low temperature for factual, consistent responses
        )
        
        # Create QA chain with sales-focused prompting
        chain = load_qa_chain(
            llm, 
            chain_type="stuff"
        )
        
        # Get answer
        with st.spinner("🔍 Searching Datacrumbs website content..."):
            # Add rate limiting
            time.sleep(1)
            
            # Always use website content and frame as sales assistant
            sales_prompt = f"""You are a helpful sales assistant for Datacrumbs. Based ONLY on the provided website content, answer the following question about Datacrumbs services, products, or company information. 

If the information is not available in the website content, politely say so and suggest contacting Datacrumbs directly.

Be professional, helpful, and sales-oriented in your response.

Question: {question}"""
            
            response = chain.run(
                input_documents=documents, 
                question=sales_prompt
            )
            
            return f"🛍️ **Datacrumbs Sales Assistant:**\n\n{response}\n\n---\n*Need more information? Contact Datacrumbs directly for detailed assistance.*"
            
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            return "⚠️ **API Quota Exceeded**\n\nOur service is temporarily unavailable due to high demand. Please try again in a few minutes."
        elif "api" in str(e).lower() and "key" in str(e).lower():
            return "🔑 **Service Configuration Error**\n\nPlease contact support - there's an issue with our API configuration."
        else:
            return f"❌ **Service Error**\n\nSorry, I'm experiencing technical difficulties. Please try again or contact Datacrumbs directly.\n\nError details: {str(e)}"

# Main application logic
if api_key:
    # Load website content if not already loaded
    if not st.session_state.website_loaded:
        with st.status("🔄 Loading Datacrumbs website...", expanded=True) as status:
            st.write("Connecting to http://www.datacrumbs.org...")
            documents = load_website_content()
            
            if documents:
                st.session_state.documents = documents
                st.session_state.website_loaded = True
                status.update(label="✅ Website content loaded successfully!", state="complete")
                
                # Show some stats about the loaded content
                total_content = " ".join([doc.page_content for doc in documents])
                word_count = len(total_content.split())
                st.info(f"📊 **Content Stats:** {len(documents)} chunks, ~{word_count:,} words scraped from Datacrumbs")
            else:
                status.update(label="❌ Failed to load website content", state="error")
                st.error("Could not scrape content from http://www.datacrumbs.org")
                st.info("The app will still work for creative questions using general AI knowledge.")
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {q[:60]}..." if len(q) > 60 else f"Q{i+1}: {q}"):
                st.write(f"**Customer:** {q}")
                st.markdown(f"**Sales Assistant:** {a}")
    
    # Question input
    st.subheader("❓ Ask About Datacrumbs")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "What would you like to know about Datacrumbs?",
            placeholder="e.g., What services do you offer? What are your prices?",
            key="question_input"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Quick question buttons - Sales focused
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏢 About Company"):
            question = "Tell me about Datacrumbs - what does the company do and what's your mission?"
            ask_button = True
    
    with col2:
        if st.button("📋 Services & Products"):
            question = "What services and products does Datacrumbs offer? What solutions do you provide?"
            ask_button = True
    
    with col3:
        if st.button("💰 Pricing & Packages"):
            question = "What are your pricing options? Do you offer different packages or plans?"
            ask_button = True
    
    with col4:
        if st.button("📞 Contact & Support"):
            question = "How can I contact Datacrumbs? What support options are available?"
            ask_button = True
    
    # Process question
    if ask_button and question:
        # Always use website content for sales assistance
        if st.session_state.documents:
            st.info("🔍 Searching Datacrumbs website for your answer...")
        else:
            st.error("⚠️ Website content not loaded - please wait for content to load first.")
        
        # Get answer (always use website content)
        answer = get_answer(question, st.session_state.documents, api_key)
        
        # Display answer
        st.markdown("---")
        st.subheader("🛍️ Sales Assistant Response")
        st.markdown(answer)
        
        # Add to chat history
        st.session_state.chat_history.append((question, answer))
        
        # Clear input
        st.session_state.question_input = ""
        
        # Rerun to update the interface
        st.experimental_rerun()

else:
    st.error("🔑 Google API Key not found!")
    st.info("The sales assistant needs an API key to function. Please set up your environment variable.")
    
    # Instructions for setting up environment variable
    with st.expander("🔧 How to set up the environment variable"):
        st.markdown("""
        **Option 1: .env file (Recommended)**
        1. Create a `.env` file in your project directory
        2. Add this line: `GOOGLE_API_KEY=your_api_key_here`
        3. Make sure to add `.env` to your `.gitignore` file
        
        **Option 2: Command Line**
        ```bash
        export GOOGLE_API_KEY="your_api_key_here"
        streamlit run app.py
        ```
        
        **Get your API key from:** [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)

# Footer
st.markdown("---")
st.markdown("**🛍️ Datacrumbs Sales Assistant** - Powered by website content scraping and AI")

# Usage instructions
with st.expander("📖 How to use the Sales Assistant"):
    st.markdown("""
    **🎯 Purpose:** This is a sales-focused chatbot that answers questions ONLY based on Datacrumbs website content.
    
    **✅ What I can help with:**
    - Company information and background
    - Services and products offered
    - Pricing and package details
    - Contact information and support
    - Technical specifications
    - Any other information found on the website
    
    **❌ What I cannot do:**
    - Answer creative or general questions
    - Provide information not on the website
    - Make up information or speculate
    
    **💡 Tips for best results:**
    - Ask specific questions about Datacrumbs services
    - Use the quick question buttons for common inquiries
    - Be direct and sales-focused in your questions
    - If I can't find the answer, I'll suggest contacting Datacrumbs directly
    """)
  
