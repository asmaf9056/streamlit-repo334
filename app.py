# app.py

import os, asyncio, nest_asyncio
nest_asyncio.apply()

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="📘 Datacrumbs Chatbot", layout="centered")
st.title("📘 Datacrumbs Info Chatbot")

@st.cache_data(show_spinner="Loading site content...")
def load_docs():
    loader = WebBaseLoader("https://datacrumbs.org")
    docs = loader.load()
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

docs = load_docs()

os.environ["GOOGLE_API_KEY"] = st.secrets["google_genai"]["google_api_key"]
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
qa_chain = load_qa_chain(llm, chain_type="stuff")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question…")
if query:
    with st.spinner("Generating response…"):
        answer = qa_chain.run(input_documents=docs, question=query)
    st.session_state.chat_history += [("user", query), ("bot", answer)]

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
