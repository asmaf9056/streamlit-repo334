import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load API key from secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.5
    )
except:
    llm = None

# Datacrumbs information
@st.cache_data
def get_datacrumbs_info():
    return """
    DATACRUMBS - COMPLETE INFORMATION

    LOCATION & CONTACT:
    Address: Room # 105, Shahrah-e-Faisal, Karachi, Pakistan
    Phone: +92 336 250 7273
    Email: help@datacrumbs.org
    Website: datacrumbs.org

    MENTOR & TEAM:
    Lead Mentor: Abis Hussain Syed
    Expert instructors with industry experience

    COURSES & PRICING:
    • Data Science Bootcamp - Rs. 29,999 (4 months)
    • Data Analytics Bootcamp - Rs. 29,999 (4 months) 
    • Business Intelligence Bootcamp - Rs. 29,999 (4 months)
    • GenAI Bootcamp (Generative AI) - Rs. 29,999
    • Ultimate Python Bootcamp - Rs. 25,000
    • SQL Zero to Hero - Rs. 15,000
    • Excel for Everyone - Rs. 12,000

    FEATURES:
    ✓ Industry-ready curriculum
    ✓ Hands-on projects
    ✓ Certification upon completion
    ✓ Internship opportunities
    ✓ Career placement assistance
    ✓ Live mentorship sessions
    ✓ 24/7 community support
    """

# Page config
st.set_page_config(page_title="Datacrumbs Chatbot", page_icon="💬")

# Simple header
st.title("Datacrumbs Chatbot")
st.subheader("I'm your virtual assistant today")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"""
You are a helpful sales assistant for Datacrumbs, an educational platform in Karachi, Pakistan.

Answer questions about courses, pricing, location, and services professionally and friendly.

DATACRUMBS INFO:
{get_datacrumbs_info()}

Keep responses conversational and helpful.
""")
    ]

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.write(message.content)

# Chat input
if prompt := st.chat_input("Ask me about Datacrumbs courses..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if llm:
            try:
                response = llm(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except:
                fallback_msg = "Please visit datacrumbs.org or contact us at help@datacrumbs.org for more information."
                st.write(fallback_msg)
                st.session_state.messages.append(AIMessage(content=fallback_msg))
        else:
            fallback_msg = "Please visit datacrumbs.org or contact us at help@datacrumbs.org for more information."
            st.write(fallback_msg)
            st.session_state.messages.append(AIMessage(content=fallback_msg))

# Simple footer
st.markdown("---")
st.markdown("**Contact:** help@datacrumbs.org | +92 336 250 7273 | [datacrumbs.org](https://datacrumbs.org)")

