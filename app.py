import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="📘 Datacrumbs Bot", layout="centered")
st.title("📘 Datacrumbs QA Chatbot")
st.caption("Ask anything about Datacrumbs.org or something creative!")

# Load and split website content
@st.cache_resource(show_spinner="Loading Datacrumbs content...")
def load_docs():
    loader = WebBaseLoader("https://datacrumbs.org")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

docs = load_docs()

# Set API key and initialize model
GEMINI_KEY = st.secrets["google_genai"]["google_api_key"]
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
qa_chain = load_qa_chain(llm, chain_type="stuff")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a question...")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(input_documents=docs, question=query)
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
