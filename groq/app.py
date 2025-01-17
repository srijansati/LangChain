import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()

    st.session_state.loader = WebBaseLoader(web_path= "https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[0:50])

    st.session_state.db = FAISS.from_documents(documents= st.session_state.documents, embedding= st.session_state.embeddings)

st.title("ChatGroq")

llm = ChatGroq(groq_api_key = groq_api_key, model_name= "mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template("""
answer the question based on the provoded context only.
please provide the best response possible based on the given question.
<context>
{context}
</context>
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm= llm, prompt= prompt)
retriever = st.session_state.db.as_retriever()

retrival_chain = create_retrieval_chain(retriever= retriever, combine_docs_chain= document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrival_chain.invoke({"input":prompt})
    print("Response Time:", time.process_time() - start)
    st.write(response["answer"])

    #writing a streamlit expander
    with st.expander("Document Similarity Search"):
        #find relevent chunk
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("____________________________________________")
 



