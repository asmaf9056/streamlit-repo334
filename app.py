import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Datacrumbs Chat",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-msg {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .bot-msg {
        background-color: #f1f8e9;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .sample-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 Chat with Datacrumbs")
st.markdown("Ask me anything about Datacrumbs or creative questions!")

# Simple API key input for testing
api_key = st.text_input("🔑 Enter your Google API Key:", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

if not api_key:
    st.warning("⚠️ Please enter your Google API Key above to start chatting.")
    st.info("Get your free API key from: https://aistudio.google.com/app/apikey")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Sample website content (simulated for speed)
DATACRUMBS_INFO = """
Datacrumbs is a data analytics platform that helps businesses transform raw data into actionable insights. 
We specialize in:
- Data visualization and dashboard creation
- Analytics consulting and strategy
- Data science training and workshops
- Business intelligence solutions

Founded by experienced data scientists, Datacrumbs serves companies of all sizes to make data-driven decisions.
Our mission is to make data analytics accessible and valuable for every organization.
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sample questions
st.markdown("""
<div class="sample-box">
    <h4>💡 Try these questions:</h4>
    <p><strong>About Datacrumbs:</strong></p>
    <ul>
        <li>What is Datacrumbs?</li>
        <li>What services do you offer?</li>
        <li>Who founded Datacrumbs?</li>
    </ul>
    <p><strong>Creative:</strong></p>
    <ul>
        <li>Write a haiku about data science</li>
        <li>Tell me a story about magic spreadsheets</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Display chat history
if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <strong>🧑 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-msg">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.text_input("💬 Your question:", placeholder="Type your message here...", key="user_input")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    try:
        # Check if question is about Datacrumbs
        if any(keyword in user_input.lower() for keyword in 
               ['datacrumbs', 'service', 'offer', 'company', 'about', 'what is', 'founded', 'team']):
            
            # Website-related response
            prompt = f"""
            Based on this information about Datacrumbs: {DATACRUMBS_INFO}
            
            Question: {user_input}
            
            Please provide a helpful answer about Datacrumbs based on the information provided.
            """
        else:
            # Creative/general response
            prompt = f"Please answer this question in a helpful and creative way: {user_input}"
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Refresh to show new messages
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please try again or rephrase your question.")

# Clear chat button
if st.button("🗑️ Clear Chat", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with Streamlit & Google Gemini | Visit: <a href="https://datacrumbs.org" target="_blank">datacrumbs.org</a>
</div>
""", unsafe_allow_html=True)
