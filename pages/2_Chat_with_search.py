import streamlit as st

from langchain.agents import initialize_agent, AgentType, Tool, load_tools, create_csv_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.callbacks import StdOutCallbackHandler
from langchain.llms import OpenAI
import pandas as pd

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI as PandasOpenAI



import os
# CODI_AUDITORIA,dbo_EMPRESAS.ENTITAT,CODI,CODI_SERVEI,Search Name,TIPUS_PROGRAMA,ESTAT_OPORTUNITAT,ESTAT_RESOLUCIO_CONVENI,ANY_CONVOCATORIA,CONVOCATORIA,CODI_PROGRAMA_,NAPR,NOM_ACCIO,CODI_SERVEI_ESPECIALITAT_UP,NUM_EXPEDIENT,ACTIVACIO_PROGRAMES,DATA_INICI_PROGRAMA,DATA_INICI_VERIFICADA,DATA_FI_PROGRAMA,DATA_FINAL_VERIFICADA,
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    dt = pd.read_excel('/workspaces/chat-bot-app/cursos.xlsx')
    csv_file = dt.to_csv('/workspaces/chat-bot-app/cursos_.csv', index=False)
    dt = pd.read_csv('./cursos_.csv')
    columns_to_drops = ['CODI_AUDITORIA', 'dbo_EMPRESAS.ENTITAT', 'CODI', 'CODI_SERVEI',
                        'CODI_SERVEI_ESPECIALITAT_UP', 'CODI_PROGRAMA_', 'NUM_EXPEDIENT', 'ACTIVACIO_PROGRAMES']
    dt.drop(columns_to_drops, inplace=True, axis=1)
    dt.rename(columns={'Search Name': 'NOM_BUSQUEDA'})
    os.environ["OPENAI_API_KEY"] = openai_api_key

    ## save dataframe to session state
    st.session_state["csv_file_dt"] = dt
    # convert to csv
    csv_file = dt.to_csv('/workspaces/chat-bot-app/cursos_.csv', index=False)
    st.write(dt)


st.title("🔎 Gentis chatbot")
st.image("./gentis5.png")

# add image to the center of the page using markdown

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
            "content": "Hola, sóc un chatbot. Com puc ajudar-te?"
         }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    df = st.session_state["csv_file_dt"]

    llm = PandasOpenAI(api_token=openai_api_key)

    pandas_ai = PandasAI(llm,conversational=True, verbose=True)



    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        language_prompt = "The response  of the user question must always be in Catalan. \n\n"
        prompt = prompt + "\n\n" + language_prompt
        response = pandas_ai(df,prompt=prompt)
        print("Pandas AI RESPONSE", response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
