import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import time

# Load Groq API key from secrets with error handling
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    api_key = None

# Set up LangChain LLM with Groq only if API key is available
if api_key:
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.5
    )
else:
    llm = None

# Enhanced scraping function for multiple pages
@st.cache_data(show_spinner="Scraping Datacrumbs website...")
def scrape_datacrumbs_site():
    """Scrape multiple pages from datacrumbs.org for comprehensive information"""
    
    pages_to_scrape = [
        "https://datacrumbs.org",
        "https://datacrumbs.org/our-courses/",
        "https://datacrumbs.org/about-us/",
        "https://datacrumbs.org/contact/",
        "https://datacrumbs.org/internship/",
        "https://datacrumbs.org/blog/"
    ]
    
    all_content = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in pages_to_scrape:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text content
                text = soup.get_text(separator="\n", strip=True)
                
                # Clean up the text
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                clean_text = '\n'.join(lines)
                
                page_name = url.split('/')[-2] if url.split('/')[-1] == '' else url.split('/')[-1]
                if not page_name:
                    page_name = "home"
                
                all_content[page_name] = clean_text[:2000]  # Limit each page
                
            time.sleep(0.5)  # Be respectful to the server
            
        except Exception as e:
            st.warning(f"Could not scrape {url}: {str(e)}")
            continue
    
    # Fallback content if scraping fails
    if not all_content:
        return """
        Datacrumbs is an educational platform offering tech courses in Pakistan.
        
        COURSES OFFERED:
        - Data Science Bootcamp (4 months) - Rs. 29,999
        - Data Analytics Bootcamp (4 months) - Rs. 29,999  
        - Business Intelligence Bootcamp (4 months)
        - GenAI Bootcamp (Generative AI)
        - Ultimate Python Bootcamp
        
        FEATURES:
        - Industry-ready programs
        - Certification provided
        - Internship opportunities
        - For college students & working professionals
        - Affordable pricing with discounts
        
        CONTACT:
        - Email: help@datacrumbs.org
        - Phone: +92 336 250 7273
        - Website: datacrumbs.org
        
        MISSION: Making tech education affordable and accessible in Pakistan
        Focus on Data Science, Python, Machine Learning, and Generative AI
        """
    
    # Combine all scraped content
    combined_content = ""
    for page, content in all_content.items():
        combined_content += f"\n=== {page.upper()} PAGE ===\n{content}\n"
    
    return combined_content[:8000]  # Limit total content

# Scrape the site
site_data = scrape_datacrumbs_site()

# Streamlit UI
st.set_page_config(
    page_title="Datacrumbs Sales Assistant", 
    page_icon="🛒",
    layout="centered"
)

# Header with logo/styling
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h1>🛒 Datacrumbs Sales Assistant</h1>
    <p style="font-style: italic; color: #666;">Your AI sales assistant for <a href="http://www.datacrumbs.org" target="_blank">datacrumbs.org</a></p>
    <p><em>Ask me anything about Datacrumbs services, products, pricing, or company information!</em></p>
</div>
""", unsafe_allow_html=True)

# Display current status
with st.sidebar:
    st.header("📊 Sales Information")
    
    st.subheader("🧠 What I can help with:")
    st.markdown("""
    • **Company Information** - About Datacrumbs
    • **Services & Products** - What we offer  
    • **Pricing & Packages** - Cost information
    • **Contact Details** - How to reach us
    • **Technical Specs** - Product features
    • **Support** - Customer service info
    """)
    
    # Model selection section
    st.subheader("🎯 Model Selection:")
    st.info("Choose Groq Model:")
    
    # Show scraped data preview
    with st.expander("📄 Website Data Preview"):
        st.text_area("Scraped Content (first 500 chars):", 
                    value=site_data[:500] + "..." if len(site_data) > 500 else site_data,
                    height=200, disabled=True)

# Main chat interface
st.subheader("💬 Datacrumbs Sales Assistant")
st.caption("Powered by Groq ⚡ for lightning-fast responses")

# Memory for chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"""
You are a professional sales assistant for Datacrumbs (datacrumbs.org) - an educational platform offering affordable tech courses in Pakistan.

Your role is to:
1. Answer questions about courses, pricing, and services
2. Help potential customers understand our offerings
3. Provide accurate contact information
4. Explain course features and benefits
5. Be helpful, friendly, and professional

Use ONLY the following website information to answer questions. If asked about something not covered in the data, politely redirect to contacting Datacrumbs directly.

Keep responses concise (under 500 characters) and helpful.

WEBSITE INFORMATION:
{site_data}

Always be positive about Datacrumbs and focus on helping customers find the right course for their needs.
""")
    ]

# Display greeting
if "greeted" not in st.session_state:
    st.success("👋 Hello! How can I help you with Datacrumbs today?")
    st.session_state.greeted = True

# User input
user_input = st.chat_input("Ask me about Datacrumbs courses, pricing, or services...")

if user_input:
    if not api_key or not llm:
        with st.chat_message("assistant"):
            st.markdown("I don't understand. Please visit our website at [datacrumbs.org](https://datacrumbs.org) for more information.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Get AI response
    with st.spinner("🤔 Thinking..."):
        try:
            response = llm(st.session_state.messages)
            ai_response = response.content.strip()
            
            # Limit response length
            if len(ai_response) > 500:
                ai_response = ai_response[:497] + "..."
            
            st.session_state.messages.append(AIMessage(content=ai_response))
            
        except Exception as e:
            with st.chat_message("assistant"):
                st.markdown("I don't understand. Please visit our website at [datacrumbs.org](https://datacrumbs.org) for more information.")

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, SystemMessage):
        continue  # Don't show system messages
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    
    with st.chat_message(role):
        st.markdown(msg.content)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9em;">
    📞 <strong>Contact Datacrumbs:</strong> help@datacrumbs.org | +92 336 250 7273<br>
    🌐 Visit: <a href="https://datacrumbs.org" target="_blank">datacrumbs.org</a>
</div>
""", unsafe_allow_html=True)
