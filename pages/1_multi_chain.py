from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import create_csv_agent;

import pinecone


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI as PandasOpenAI








import pandas as pd


st.set_page_config(layout="wide")


PINECONE_API_KEY = "a90acfd0-dd09-4eed-a2d4-4ee5739bfcea"  # find at app.pinecone.io
PINECONE_ENV = "northamerica-northeast1-gcp"  # next to api key in console
INDEX_NAME = "openai-youtube-transcriptions"

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    #dt = pd.read_excel('/workspaces/chat-bot-app/cursos.xlsx');
    #csv_file = dt.to_csv('/workspaces/chat-bot-app/cursos_.csv', index=False)
    #st.write(dt)
    #st.session_state["csv_file"] = dt
    pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
    )


st.title("🔎 Gentis chatbot Documents")
st.image("./gentis5.png")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
            "content": "Hola, sóc un chatbot. Com puc ajudar-te?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    csv = CSVLoader(file_path='/workspaces/chat-bot-app/cursos_.csv').load()
    cursos_docs_csv = FAISS.from_documents(csv, OpenAIEmbeddings()).as_retriever()

    ## pinecone retreaver here

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(csv)

    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_documents(docs, embeddings, index_name=INDEX_NAME).as_retriever()


    retriever_infos = [
        {
            "name": "about youtube transcriptions",
            "description": "good for answering questions about youtube videos transcriptions",
            "retriever": docsearch
        }
    ]


    chain = MultiRetrievalQAChain.from_retrievers(OpenAI(temperature=0), retriever_infos, verbose=True)
    
    llm  = PandasOpenAI(api_token=os.environ["OPENAI_API_KEY"])

    

    #csv agent

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        print("CHAIIIIIN---->", cursos_docs_csv)
        response = chain.run(
            prompt, callbacks=[st_cb])
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
