import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
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
api_key = os.getenv('GROQ_API_KEY')

# Sidebar for information
with st.sidebar:
    st.header("📊 Sales Information")
    
    if api_key:
        st.success("Sales Assistant Ready ✅")
        st.info("Powered by Groq ⚡")
    else:
        st.error("Groq API Key not found in environment variables")
        st.markdown("Please set GROQ_API_KEY environment variable")
    
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
    
    st.markdown("---")
    st.markdown("### 🚀 Model Selection:")
    model_choice = st.selectbox(
        "Choose Groq Model:",
        [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        index=0,
        help="Different models offer varying speed/quality tradeoffs"
    )

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
                chunk_size=2000,  # Groq models can handle larger chunks
                chunk_overlap=200,
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

def get_answer(question: str, documents: List[Document], api_key: str, model: str):
    """Get answer using LangChain and Groq - Website content only"""
    try:
        # Check if we have website content
        if not documents or len(documents) == 0:
            return "❌ **Website Content Not Available**\n\nI need to load the Datacrumbs website content first to answer your questions. Please wait for the content to load or refresh the page."
        
        # Initialize the Groq LLM with settings optimized for sales
        llm = ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=0.2,  # Low temperature for factual, consistent responses
            max_tokens=1024,  # Reasonable response length
            timeout=30,  # 30 second timeout
            max_retries=2
        )
        
        # Create QA chain with sales-focused prompting
        chain = load_qa_chain(
            llm, 
            chain_type="stuff"
        )
        
        # Get answer
        with st.spinner("🚀 Processing with Groq (Lightning Fast!)..."):
            # Groq is fast, minimal delay needed
            time.sleep(0.2)
            
            # Always use website content and frame as sales assistant
            sales_prompt = f"""You are a helpful sales assistant for Datacrumbs. Based ONLY on the provided website content, answer the following question about Datacrumbs services, products, or company information. 

Key instructions:
- Only use information from the provided website content
- If information is not available in the content, politely say so and suggest contacting Datacrumbs directly
- Be professional, helpful, and sales-oriented in your response
- Provide specific details when available
- Format your response clearly and professionally

Question: {question}"""
            
            response = chain.run(
                input_documents=documents, 
                question=sales_prompt
            )
            
            return f"🛍️ **Datacrumbs Sales Assistant (via Groq ⚡):**\n\n{response}\n\n---\n*Need more information? Contact Datacrumbs directly for detailed assistance.*"
            
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower() or "rate" in str(e).lower():
            return "⚠️ **API Rate Limit Reached**\n\nGroq API rate limit exceeded. Please wait a moment and try again."
        elif "api" in str(e).lower() and "key" in str(e).lower():
            return "🔑 **Service Configuration Error**\n\nPlease contact support - there's an issue with our Groq API configuration."
        elif "timeout" in str(e).lower():
            return "⏱️ **Request Timeout**\n\nThe request took too long. Please try a simpler question or try again."
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
    
    # Question input using proper form structure
    st.subheader("❓ Ask About Datacrumbs")
    
    # Create a form for the question input to prevent unwanted reruns
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "What would you like to know about Datacrumbs?",
            placeholder="e.g., What services do you offer? What are your prices?",
            key="question_input"
        )
        
        # Submit button inside the form
        ask_button = st.form_submit_button("Ask ⚡", type="primary", use_container_width=True)
    
    # Quick question buttons - Sales focused
    st.markdown("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_question = None
    
    with col1:
        if st.button("🏢 About Company", key="about_btn"):
            quick_question = "Tell me about Datacrumbs - what does the company do and what's your mission?"
    
    with col2:
        if st.button("📋 Services & Products", key="services_btn"):
            quick_question = "What services and products does Datacrumbs offer? What solutions do you provide?"
    
    with col3:
        if st.button("💰 Pricing & Packages", key="pricing_btn"):
            quick_question = "What are your pricing options? Do you offer different packages or plans?"
    
    with col4:
        if st.button("📞 Contact & Support", key="contact_btn"):
            quick_question = "How can I contact Datacrumbs? What support options are available?"
    
    # Process question from form submission or quick buttons
    current_question = question if ask_button else quick_question
    
    if current_question:
        # Always use website content for sales assistance
        if st.session_state.documents:
            st.info("🚀 Processing with Groq - Lightning fast responses!")
        else:
            st.error("⚠️ Website content not loaded - please wait for content to load first.")
        
        # Get answer (always use website content)
        answer = get_answer(current_question, st.session_state.documents, api_key, model_choice)
        
        # Display answer
        st.markdown("---")
        st.subheader("🛍️ Sales Assistant Response")
        st.markdown(answer)
        
        # Add to chat history
        st.session_state.chat_history.append((current_question, answer))
        
        # Rerun to update the interface
        st.rerun()

else:
    st.error("🔑 Groq API Key not found!")
    st.info("The sales assistant needs a Groq API key to function. Please set up your environment variable.")
    
    # Instructions for setting up environment variable
    with st.expander("🔧 How to set up the Groq API Key"):
        st.markdown("""
        **Option 1: .env file (Recommended)**
        1. Create a `.env` file in your project directory
        2. Add this line: `GROQ_API_KEY=your_groq_api_key_here`
        3. Make sure to add `.env` to your `.gitignore` file
        
        **Option 2: Command Line**
        ```bash
        export GROQ_API_KEY="your_groq_api_key_here"
        streamlit run app.py
        ```
        
        **Get your Groq API key from:** [Groq Console](https://console.groq.com/keys)
        
        **Why Groq?**
        - ⚡ Ultra-fast inference speeds
        - 🎯 High-quality responses
        - 💰 Competitive pricing
        - 🚀 Optimized for production workloads
        """)

# Footer
st.markdown("---")
st.markdown("**🛍️ Datacrumbs Sales Assistant** - Powered by Groq ⚡ for lightning-fast responses")

# Usage instructions
with st.expander("📖 How to use the Sales Assistant"):
    st.markdown("""
    **🎯 Purpose:** This is a sales-focused chatbot that answers questions ONLY based on Datacrumbs website content.
    
    **⚡ Powered by Groq:** Ultra-fast AI inference for instant responses!
    
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
    
    **🚀 Model Information:**
    - **llama3-8b-8192**: Fast, efficient, good for most queries
    - **llama3-70b-8192**: More powerful, better for complex questions
    - **mixtral-8x7b-32768**: Large context window, good for detailed analysis
    - **gemma-7b-it**: Optimized for instruction following
    """)
