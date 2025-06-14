import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="📘 Datacrumbs Info Chatbot", layout="centered")
st.title("📘 Datacrumbs Info Chatbot")
st.caption("Ask me anything about Datacrumbs.org or even something creative!")

# Load documents from Datacrumbs.org
@st.cache_resource(show_spinner="Loading website content...")
def load_documents():
    loader = WebBaseLoader("https://datacrumbs.org")
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(raw_docs)

docs = load_documents()

# Load Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.chat_input("Ask a question about Datacrumbs or anything else...")

if query:
    result = qa_chain.run(input_documents=docs, question=query)
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", result))

# Chat display
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
