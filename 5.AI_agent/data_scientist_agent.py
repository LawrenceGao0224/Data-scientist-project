import pandas as pd
csv_file='power consumption.csv'
document = pd.read_csv(csv_file)

#Create AI Agent
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
ds_assist = create_pandas_dataframe_agent(
    llm,
    document,
    allow_dangerous_code=True,
    verbose=True
)

ds_assist.invoke("Analyze this data, and write a brief explanation around 100 words.")