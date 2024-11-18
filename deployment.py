from langchain_openai import AzureChatOpenAI, AzureOpenAI
from streamlit_extras.stylable_container import stylable_container
import json
import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import subprocess
import base64
import datetime
import io
import re
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from pyprojroot import here
import warnings
warnings.filterwarnings("ignore")
import dotenv
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,PromptTemplate

# Streamlit UI for chat
st.set_page_config(layout="wide")
st.title(":blue[DataStellar]")
st.markdown(":grey[_You partner in analytics_]")

def get_session_state(**kwargs):
	for key, val in kwargs.items():
		if key not in st.session_state:
			st.session_state[key] = val
	return st.session_state
session_state = get_session_state(title="")
session_state =  get_session_state(display_history_flag = "No")

if session_state.title:
	# Generate a unique file name based on current timestamp
	if "conversation_file" not in st.session_state:
		# timestamp = datetime.datetime.now().strftime("%H%M%S")
		# unique_file_name = str(session_state.title)+"_"+str(timestamp)
		# st.markdown(unique_file_name)
		unique_file_name = session_state.title
		conversation_files = os.listdir(r"C:\Users\SNivassankaramoorthy\vscode\NeuralNetworkNinjas-1\chat_history")
		if conversation_files:
			for file in conversation_files:
				if file == session_state.title:
					timestamp = datetime.datetime.now().strftime("%H%M%S")
					unique_file_name = str(session_state.title)+"___"+str(timestamp)
		conversation_file = os.path.join(r"C:\Users\SNivassankaramoorthy\vscode\NeuralNetworkNinjas-1\chat_history", unique_file_name)
		st.session_state["conversation_file"] = conversation_file

def save_conversation(conversation, conversation_file):
	with open(conversation_file, 'w') as json_file:
		json.dump(conversation, json_file)

# Initialize conversation history
if 'conversation_history' not in st.session_state:
	st.session_state.conversation_history = []

# Initialize a list to store image data and messages
if 'image_data_list' not in st.session_state:
	st.session_state.image_data_list = []

dotenv.load_dotenv()
print("Environment variables are loaded:", load_dotenv())

#Connect SQLDatabase
db_path = "data/sqldb"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

#Load LLM
# configure Azure OpenAI service client
llm = AzureChatOpenAI(
	azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
	api_key=os.environ['AZURE_OPENAI_API_KEY'],  
	api_version = os.environ['AZURE_OPENAI_API_VERSION'],
	temperature=0.0
  )

VISUAL_PROMPT = """
You are a skilled data analyst with a strong understanding of data visualization techniques.
 
You will always get a dataset as input from the user. Your task is to analyze the given dataset and suggest the most suitable Plotly visualization to highlight the key insights.
 
Consider the following steps before sending the response:
1. A concise explanation of the recommended Plotly visualization technique.
2. **Python code to create a DataFrame from the given dataset**
3. **Within the python code, include the required Plotly libraries; apply the recommended Plotly visualization chart on the given dataset; and save the output chart as an HTML file with the default file name 'output_visualization.html'**
4. **Make sure the generated Python code can be executed without any additional changes.**
5. Give the complete Python code to the user, ensuring correct syntax and indentation and no need to show the image in python code.
 
Consider the following factors:
-Beautify: every time please generate different color and charts.
- Data Types: Analyze the data types of the columns in the given dataset to determine appropriate Plotly visualizations.
- Data Distribution: Consider the distribution of the data in the given dataset to choose suitable Plotly visualizations.
- Relationships: Identify relationships between variables in the given dataset and select Plotly visualizations that highlight these relationships.
- Clarity and Conciseness: Ensure the Plotly visualization is clear, concise, and easy to interpret.
 
**If the given dataset is empty or has only one column or is simple, there's no need to generate any Plotly visualization charts. Just respond to the user politely that a Plotly visualization is not required.**
 
**Response Format:**
 
Data Visualization: {recommended Plotly visualization technique}
 
Python Code for Visualization: {python_code}
 
"""

def get_visualization_python_response(prompt):
	messages = [
		{"role": "system", "content": VISUAL_PROMPT}
	]

	messages.append({"role": "user", "content": prompt})
	
	response = llm.invoke(messages)

	return response.content

def extract_python_code(response):
	python_code_match = re.search(r"```python\n(.*?)\n```", response, flags=re.DOTALL)
	if python_code_match:
		return python_code_match.group(1)
	else:
		return None
		
def run_python_code_and_return_image(code_string):
	with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
		temp_file.write(code_string.encode('utf-8'))

	try:
		subprocess.run([os.environ['PATH'], temp_file.name])
		# image_path = 'output_image.png'
		# with open(image_path, 'rb') as f:
		# 	image_bytes = f.read()

		# base64_image = base64.b64encode(image_bytes).decode('utf-8')
		# return base64_image
	finally:
		os.remove(temp_file.name)

#Answer Prompt Template
answer_prompt = PromptTemplate.from_template(
	"""Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}

consider the following points before answering.
1. **If user question is generic like (feedback/casual talk/greetings/appreciation/general questions), respond to their queries** No need to consider the SQL Query and SQL Result. 
2. **If user question is related to query on database (SQL query) and the SQL result is a valid dataset with more than one record, then answer to the user question in tabular format.**
3. If user question is related to query on data (SQL query) and the SQL result is empty or having single record with only one column, then give text response answer to the user.

key points: 
if user question is like perfect/good/awesome/beautiful/wonderful/amazing/fantastic means, its a sign of they appreciating your performance. for this type of user questions, No need to consider the SQL Query and SQL Result.

Answer: """
)
answer = answer_prompt | llm | StrOutputParser()

final_prompt = ChatPromptTemplate.from_messages(
	[
		("system", "You are a SQL expert. Given an input question, conversation history and metadata {metadata}, Use the metadata file to understand the tables, columns and it relationships; and create a syntactically correct SQL query to run. Make sure date/datetime columns are handled using strftime instead of DATE_FORMAT. share me only sql query."),
		MessagesPlaceholder(variable_name = "messages"),
		("human","{input}"),
		("system", "**Top K:** {top_k}"),
		("system", "**Table Information:** {table_info}")
	]
)
write_query = create_sql_query_chain(llm, db, final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)

def user_query_categoization(user_input):
	initial_prompt = PromptTemplate.from_template("""1. Your name is 'Datastellar', in case if user asks about your name please keep this in mind. 
							   2. In case if user asks about your role,  summarize like, you are an AI powered Data exploration and Visualization Agent. you are expert in SQL and you help users in analysing the data and also giving insights to user by generating data visualization. summarise this in a better way to user.
							   please note the above two points is only for user to understand who you are, in case if they ask about you. But your actual role is below only.
							   Classify the user input based on the intent: User input: {user_input} \n\n
							   Classify this as either 'Analytics' if it's a SQL or data related query. \n\n
							   Classify this as 'Generic' if it is not SQL or not data related query. 
							   **if the user asks to modify or delete records in any table; or drop any tables and dbs (anything other than SELECT query). Respond to the user politely saying you dont have permissions other than select query operations. You should treat this type of queries as 'Generic' only** \n\n
							   Category: [Category]\nResponse:[Response]
							   If the Category is not related to analytics, generate the response for actual user input
							   """
							   )
	category_prompt = initial_prompt | llm | StrOutputParser()
	categorize_chain = category_prompt.invoke(user_input)
	return categorize_chain

def clean_sql_query(data): 
	cleaned_query = data.replace("```sql", "").replace("```", "").strip() 
	return cleaned_query

def create_history(messages):
	history = ChatMessageHistory()
	if history:  # Check if history is not empty
		for turn in messages:
			if turn['role'] == 'user':
				history.add_user_message(turn['content'])
			elif turn['role'] == 'assistant' and "Output Image" not in line["content"]:
				history.add_ai_message(turn['content'])
	return history



#Display chat message from history on app rerun
for line in st.session_state.conversation_history:
	print(line)
	if line["role"] == "user":
		with st.chat_message("user"):
			st.markdown(line["content"])
	elif line["role"] == "assistant" and "Output Image" not in line["content"]:
		with st.chat_message("assistant"):
			st.markdown(line["content"])
	elif line["role"] == "assistant" and "Output Image" in line["content"]:
		with st.chat_message("assistant"):
			st.markdown("Here's the visualization:")
			html_content = line["content"].split('NeuralNetworkNinjas')[1]
			components.html(html_content, height=600, scrolling=True)

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.title(":blue[**DataStellar**]")
st.sidebar.caption("*AI assistance for analytics and reporting*")
st.sidebar.header("", divider="blue")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")

# Function to start a new conversation
def start_new_conversation():
	session_state.title = ""
	session_state.display_history_flag = "No"
	session_state.conversation_history = []  # Clear conversation history
	if 'conversation_file' in st.session_state:
		del st.session_state["conversation_file"]

# Function to clear current chat history
def clear_chat_history():
	session_state.conversation_history = []
	# Option to keep AI conversation history for context
	session_state.conversation_history = [line for line in session_state.conversation_history if line.startswith("AI: ")]

if session_state.title or session_state.display_history_flag == 'Yes':
	if 1 == 1:
		pass   
	# Create a horizontal layout using columns for buttons
	col1, col2 = st.sidebar.columns(2)
	# Add "Start Conversation" Button in the first column
	with col1:
		# Create buttons with st.button
		with stylable_container(
			"green",
			css_styles="""
			button {
				background-color: #3b5998;
				color: white;
			}""",
		):
			if st.button("Start a new Conversation", key="startconv", on_click=start_new_conversation):
		# if st.button("Start a Conversation", on_click=start_new_conversation):
		# if button1_clicked:
		#     start_new_conversation()
				pass  # Prevent running the function twice

st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")

# Add the activity tab
def load_conversation(file_path):
	with open(file_path, 'r') as json_file:
		conversation = json.load(json_file)
	return conversation

def display_conversation(conversation,selected_file):
	session_state.title=selected_file
	session_state.display_history_flag = 'Yes'
	st.title(selected_file)
	st.session_state.conversation_history.clear()
	# st.session_state.conversation_history.append(f"AI: {response}")
	for line in conversation:
		if line["role"]=="user":
			with st.chat_message("user"):
				#st.session_state.conversation_history.append(f"{line["content"]}")
				st.markdown(line["content"])
			#user_message = st.chat_message("User")
			#user_message.write(line["content"])
			#st.chat_message("user", line["content"], is_user=True)
		elif line["role"] == "assistant" and "Output Image" not in line["content"]:
			with st.chat_message("assistant"):
				st.markdown(line["content"])
		elif line["role"] == "assistant" and "Output Image" in line["content"]:
			with st.chat_message("assistant"):
				st.markdown("Here's the visualization:")
				html_content = line["content"].split('NeuralNetworkNinjas')[1]
				components.html(html_content, height=600, scrolling=True)


def file_modified_time(file_path):
	return os.path.getmtime(file_path)
# Display sorted list of conversation files
conversation_files_path = r"C:\Users\SNivassankaramoorthy\vscode\NeuralNetworkNinjas-1\chat_history"
conversation_files = sorted(os.listdir(conversation_files_path), key=lambda x: file_modified_time(os.path.join(conversation_files_path, x)), reverse=True)
if conversation_files:
	max_file_length = max(len(file) for file in conversation_files)
else:
	max_file_length = 300  # Static width value in pixels
# Set the width of the sidebar
st.markdown(f'<style>.sidebar .sidebar-content {{ width: {max_file_length}px }}</style>', unsafe_allow_html=True)
with st.sidebar.expander("History", expanded=False):
	with st.form(key='conversation_form'):
		selected_file = st.radio("Select a conversation file", conversation_files)
		print(selected_file)
		if st.form_submit_button("Load Conversation"):
			file_path = os.path.join(conversation_files_path, selected_file)
			st.session_state["conversation_file"]=file_path
			conversation = load_conversation(file_path)
			# Display the conversation content in the main bar/frame

if session_state.display_history_flag == 'Yes':
	try:
		display_conversation(conversation,selected_file)
	except NameError:
		pass
else:
	try:
		display_conversation(conversation,selected_file)
	except NameError:
		pass

	# Display the title input box
if not session_state.title:
	# Adjusted gradient color effect for the greeting text using Markdown and CSS
	st.markdown(
		f"""
		<div style="display: flex; justify-content: center;">
			<h1 style="text-align: center; font-size: 36px; background: blue; -webkit-background-clip: text; ">
				Hello, Welcome!
			</h1>
		</div>
		""",
		unsafe_allow_html=True
	)
	# Small, single-line text input box
	# st.markdown("<h7 style='font-style: italic; text-align: center;'>Name your chat to maintain your history</h7>", unsafe_allow_html=True)
	# Single-line text input box that dynamically adjusts to input length
	session_state.title = st.text_input("Name your chat to maintain your history",
		session_state.title,
		max_chars=None  # Allowing unlimited characters
	)

			   
user_input = st.chat_input("Ask a question related to data, querying, data warehousing, and analysis")

with open('metadata.json', 'r') as file:
	metadata = json.load(file)

if user_input:
	# Append user input to conversation history
	st.session_state.conversation_history.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)
		
	with st.spinner("Generating Response..."):
		with st.chat_message("assistant"):
			history = create_history(st.session_state.conversation_history)
			category_response = user_query_categoization(history.messages)
			print(category_response)
			if 'Analytics' in category_response.split(':')[1]:
				response = write_query.invoke({"question": user_input, "messages": history.messages, "metadata": metadata})
				print(response)

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
				if query_result:
					visualization_python_code = get_visualization_python_response(query_result)
					print(visualization_python_code)
					if "import" in visualization_python_code:
						visualization_code = extract_python_code(visualization_python_code)
						print(visualization_code)
						visualization_output = run_python_code_and_return_image(visualization_code)
						path_to_html = "output_visualization.html"
						with open(path_to_html, 'r', encoding='utf-8') as file:
							html_content = file.read()    
						#st.session_state.image_data_list.append(("Here's the visualization:", html_content))
						st.session_state.conversation_history.append({"role": "assistant", "content": f"AI: Output Image NeuralNetworkNinjas {html_content}"})
						st.markdown("Here's the visualization:")
						components.html(html_content, height=600, scrolling=True)
			else:
				st.session_state.conversation_history.append({"role": "assistant", "content": category_response.split(':')[2]})
				st.markdown(category_response.split(':')[2])
				#print(category_response)
	save_conversation(st.session_state.conversation_history, st.session_state["conversation_file"])
