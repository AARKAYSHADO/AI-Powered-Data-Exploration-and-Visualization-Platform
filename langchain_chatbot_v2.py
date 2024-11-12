#import openai
import streamlit as st
import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from pyprojroot import here
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,PromptTemplate


# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

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

#Answer Prompt Template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}

consider the following points before answering.
1. **If user question is generic like (feedback/casual talk/greetings/appreciation/general questions), respond to their queries** No need to consider the SQL Query and SQL Result. 
2. If user question is related to query on data (SQL query) and the SQL result is a valid dataset, then answer to the user question in tabular format.
3. If user question is related to query on data (SQL query) and the SQL result is empty, then give text response answer to the user.

key points: 
if user question is like perfect/good/awesome/beautiful/wonderful/amazing/fantastic means, its a sign of they appreciating your performance. for this type of user questions, No need to consider the SQL Query and SQL Result.

Answer: """
)
answer = answer_prompt | llm | StrOutputParser()

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a SQL expert. Given an input question and conversation history, create a syntactically correct SQL query to run. Make sure date/datetime columns are handled using strftime instead of DATE_FORMAT"),
        MessagesPlaceholder(variable_name = "messages"),
        ("human","{input}"),
        ("system", "**Top K:** {top_k}"),
        ("system", "**Table Information:** {table_info}")
    ]
)
write_query = create_sql_query_chain(llm, db, final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)

def clean_sql_query(data): 
	cleaned_query = data.replace("```sql", "").replace("```", "").strip() 
	return cleaned_query

def create_history(messages):
    history = ChatMessageHistory()
    if history:  # Check if history is not empty
        for turn in messages:
            if turn['role'] == 'user':
                history.add_user_message(turn['content'])
            elif turn['role'] == 'assistant':
                history.add_ai_message(turn['content'])
    return history

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
			history = create_history(st.session_state.conversation_history)
			response = write_query.invoke({"question": user_input, "messages": history.messages})

			# Clean the SQL query 
			query = clean_sql_query(response)
			print(query)
			
			# Execute the cleaned SQL query 
			query_result = execute_query.run(query)
			print(query_result)
			
			# Answer the user question based on SQL result 
			final_answer = answer.invoke({ "question": user_input, "query": query, "result": query_result })
			
			st.markdown(final_answer)
	st.session_state.conversation_history.append({"role": "assistant", "content": final_answer})