#import openai
import streamlit as st
import os
import re
import tempfile
import subprocess
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from pyprojroot import here
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

print("Environment variables are loaded:", load_dotenv())
api_key = os.getenv("API_KEY")

#Connect SQLDatabase
db_path = "data/sqldb"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

#Load LLM
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.0
)

#Create Agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize a list to store image data and messages
if 'image_data_list' not in st.session_state:
    st.session_state.image_data_list = []

# Streamlit UI for chat
st.title("SimpliAsk")
st.markdown("You partner in analytics")

#Display chat message from history on app rerun
for message in  st.session_state.conversation_history:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

user_input = st.chat_input("Ask a question related to data, querying, data warehousing, and analysis")

if user_input:
	# Append user input to conversation history
	st.session_state.conversation_history.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)
		
	with st.spinner("Generating Response..."):
		with st.chat_message("assistant"):
			response = agent_executor.invoke(
				{
					"input": user_input,
					"history": st.session_state.conversation_history
				}
			)
			response = response["output"]
			st.markdown(response)
	st.session_state.conversation_history.append({"role": "assistant", "content": response})