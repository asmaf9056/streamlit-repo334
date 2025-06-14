# app.py
import os
import streamlit as st
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

nest_asyncio.apply()
st.set_page_config(page_title="📘 Datacrumbs Info Chatbot")
st.title("📘 Datacrumbs Info Chatbot")
st.caption("Ask about Datacrumbs.org — or get creative!")

@st.cache_data
def load_docs():
    loader = WebBaseLoader("https://datacrumbs.org")
    data = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(data)

docs = load_docs()

os.environ["GOOGLE_API_KEY"] = st.secrets["google_genai"]["google_api_key"]
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
qa_chain = load_qa_chain(llm, chain_type="stuff")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask anything...")
if query:
    answer = qa_chain.run(input_documents=docs, question=query)
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("assistant", answer))

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)
