import streamlit as st
import asyncio
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# Fix event loop issues
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

# Set Streamlit UI
st.set_page_config(page_title="📘 Datacrumbs Chatbot")
st.title("📘 Datacrumbs Info Chatbot")
st.write("Ask me anything about Datacrumbs.org or even something creative!")

# Load documents from the site
@st.cache_data(show_spinner="Scraping website...")
def load_documents():
    loader = WebBaseLoader("https://datacrumbs.org/")
    raw_docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(raw_docs)

docs = load_documents()

# Set up Gemini Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=st.secrets["google_genai"]["google_api_key"],
    temperature=0.5,
)

# Define QA Chain
prompt_template = """
You are a helpful assistant that answers questions using content from Datacrumbs.org. 
If the answer is not in the content, reply creatively.

Question: {question}
Context: {context}
Answer:"""

prompt = PromptTemplate.from_template(prompt_template)
chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Get user query
query = st.chat_input("Ask a question...")

if query:
    with st.spinner("Thinking..."):
        response = chain.run(input_documents=docs, question=query)
        st.chat_message("user").markdown(query)
        st.chat_message("assistant").markdown(response)
