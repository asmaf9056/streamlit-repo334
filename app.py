import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load Groq API key from secrets
api_key = st.secrets["GROQ_API_KEY"]

# Set up LangChain LLM with Groq
llm = ChatOpenAI(
    model="llama3-70b-8192",
    openai_api_key=api_key,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.5
)

# Scrape datacrumbs.org
@st.cache_data(show_spinner="Scraping Datacrumbs website...")
def scrape_site():
    try:
        res = requests.get("https://datacrumbs.org", timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[:5000]  # limit for performance
    except:
        return "Datacrumbs is an educational site offering tech courses in Python, ML, and GenAI."

site_data = scrape_site()

# Streamlit UI
st.set_page_config(page_title="Datacrumbs Chatbot", layout="centered")
st.title("Welcome to Datacrumbs")

# Memory for chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"""
You are a helpful chatbot for datacrumbs.org — a website offering Python, ML, and GenAI courses for students in Pakistan.

Your job is to answer questions about the site using ONLY the following context. If unsure, say "I'm here to help with Datacrumbs-related questions only." Limit all answers to 600 characters max.

Website info:
{site_data}
""")
    ]

# Display greeting once
if "greeted" not in st.session_state:
    st.info("👋 Hello. How can I assist you today?")
    st.session_state.greeted = True

# User chat input
user_input = st.chat_input("Ask something about Datacrumbs:")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("Thinking..."):
        try:
            response = llm(st.session_state.messages)
            trimmed_reply = response.content.strip()[:600]
            st.session_state.messages.append(AIMessage(content=trimmed_reply))
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

# Display chat history
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user" if isinstance(msg, HumanMessage) else "system"
    if role != "system":  # Don't show system instructions
        with st.chat_message(role):
            st.markdown(msg.content)
