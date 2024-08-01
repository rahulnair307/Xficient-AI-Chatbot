from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader
import pickle
import streamlit as st
import os

import numpy as np
import pandas as pd
from indexer import IndexerClass

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pymilvus import MilvusClient, DataType

parent_directory = '../bitcamp/'
file_name = 'Cluster_data.csv'
file_path = parent_directory + file_name

os.environ['OPENAI_API_KEY'] = '***************************************'

df = pd.read_csv('Cluster_data.csv')

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

st.set_page_config(layout = "wide")
with st. sidebar:
    DOCS_DIR = os. path. abspath("•/uploaded_ _docs") 
    if not os. path.exists(DOCS_DIR):
        os. makedirs(DOCS_DIR)
    st. subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:",
accept_multiple_files = True)
        submitted = st. form_submit_button("Upload!")

    if uploaded_files and submitted:  
        for uploaded_file in uploaded_files:
            st. success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path. join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

st.markdown('<h1 style="font-size: 60px; text-align: center;">Xficient Bot</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 25px; margin-top: -35px; margin-bottom: 5px; text-align: center;"><em>Efficiency at its finest</em></h2>', unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Memory and chaining
prompt = st.chat_input("What is up?")
pTemplate = PromptTemplate(input_variables = ['query'], template='{query}')

bufferMemory = ConversationBufferMemory(input_key='query', memory_key='chat_history')
llm = OpenAI(temperature = 0.9)

chain = LLMChain(llm=llm, prompt=pTemplate, verbose=True, output_key='query', memory=bufferMemory)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="static/image.png"):
        stream = agent.run(prompt)
        response = st.write(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


