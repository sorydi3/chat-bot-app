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
import os
# CODI_AUDITORIA,dbo_EMPRESAS.ENTITAT,CODI,CODI_SERVEI,Search Name,TIPUS_PROGRAMA,ESTAT_OPORTUNITAT,ESTAT_RESOLUCIO_CONVENI,ANY_CONVOCATORIA,CONVOCATORIA,CODI_PROGRAMA_,NAPR,NOM_ACCIO,CODI_SERVEI_ESPECIALITAT_UP,NUM_EXPEDIENT,ACTIVACIO_PROGRAMES,DATA_INICI_PROGRAMA,DATA_INICI_VERIFICADA,DATA_FI_PROGRAMA,DATA_FINAL_VERIFICADA,
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    dt = pd.read_excel('./cursos.xlsx')
    csv_file = dt.to_csv('./cursos_.csv', index=False)
    dt = pd.read_csv('./cursos_.csv')
    columns_to_drops = ['CODI_AUDITORIA', 'dbo_EMPRESAS.ENTITAT', 'CODI', 'CODI_SERVEI',
                        'CODI_SERVEI_ESPECIALITAT_UP', 'CODI_PROGRAMA_', 'NUM_EXPEDIENT', 'ACTIVACIO_PROGRAMES']
    dt.drop(columns_to_drops, inplace=True, axis=1)
    dt.rename(columns={'Search Name': 'NOM_BUSQUEDA'})
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # convert to csv
    csv_file = dt.to_csv('./cursos_.csv', index=False)
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

    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")

    agent_csv = create_csv_agent(
        OpenAI(temperature=0),
        "cursos_.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    tools = load_tools(['ddg-search'])

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    verbose=True, handle_parsing_errors="CHECK YOUR INPUT", max_iterations=10)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        response = agent_csv.run(
            prompt, callbacks=[st_cb])
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
