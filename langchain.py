import os


import numpy as np
import pandas as pd
import DataStructurization as data

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent

df = pd.read_csv('Cluster_data.csv')

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

agent.run()

