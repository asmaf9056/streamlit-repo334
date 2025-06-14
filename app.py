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

# Fast, lightweight scraping function
@st.cache_data(show_spinner="Loading Datacrumbs info...")
def scrape_datacrumbs_site():
    """Quick scraping with pre-loaded fallback data"""
    
    # Try to scrape just the main page quickly
    try:
        response = requests.get("https://datacrumbs.org", timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Quick text extraction
            text = soup.get_text(separator=" ", strip=True)[:2000]
        else:
            text = ""
    except:
        text = ""
    
    # Comprehensive pre-loaded data (no need to scrape everything)
    base_info = """
    DATACRUMBS - COMPLETE INFORMATION

    LOCATION & CONTACT:
    Address: Room # 105, Shahrah-e-Faisal, Karachi, Pakistan
    Phone: +92 336 250 7273
    Email: help@datacrumbs.org
    Website: datacrumbs.org

    MENTOR & TEAM:
    Lead Mentor: Abis Hussain Syed
    Expert instructors with industry experience
    Personalized guidance and mentorship

    COURSES & PRICING:
    • Data Science Bootcamp - Rs. 29,999 (4 months)
    • Data Analytics Bootcamp - Rs. 29,999 (4 months) 
    • Business Intelligence Bootcamp - Rs. 29,999 (4 months)
    • GenAI Bootcamp (Generative AI) - Rs. 29,999
    • Ultimate Python Bootcamp - Rs. 25,000
    • SQL Zero to Hero - Rs. 15,000
    • Excel for Everyone - Rs. 12,000

    COURSE FEATURES:
    ✓ Industry-ready curriculum
    ✓ Hands-on projects
    ✓ Certification upon completion
    ✓ Internship opportunities
    ✓ Career placement assistance
    ✓ Live mentorship sessions
    ✓ 24/7 community support

    TARGET AUDIENCE:
    - College students
    - Working professionals
    - Career switchers
    - Fresh graduates

    SPECIALIZATIONS:
    - Data Science & Analytics
    - Machine Learning & AI
    - Python Programming
    - SQL & Database Management
    - Business Intelligence & Power BI
    - Excel Analytics
    - Generative AI & ChatGPT

    WHY CHOOSE DATACRUMBS:
    - Affordable pricing with payment plans
    - Industry-focused curriculum
    - Expert mentorship by Abis Hussain Syed
    - Practical hands-on learning
    - Job placement support
    - Based in Karachi, Pakistan
    - Flexible learning schedules
    """
    
    # Combine scraped content with base info
    if text:
        return f"{base_info}\n\nLIVE WEBSITE DATA:\n{text}"
    else:
        return base_info

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
You are a professional sales assistant for Datacrumbs (datacrumbs.org) - an educational platform in Karachi, Pakistan offering affordable tech courses.

Your role: Answer questions about courses, pricing, mentor, location, and services. Be helpful, friendly, and professional.

Keep responses SHORT (under 300 characters) and conversational.

DATACRUMBS INFO:
{site_data}

Key Details:
- Location: Room # 105, Shahrah-e-Faisal, Karachi
- Mentor: Abis Hussain Syed  
- Phone: +92 336 250 7273
- Email: help@datacrumbs.org
- Main courses: Data Science, Analytics, Python, AI (Rs. 29,999 each)

Be positive and help customers find the right course.
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
