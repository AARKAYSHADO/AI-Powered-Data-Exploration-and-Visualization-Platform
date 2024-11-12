import openai
import streamlit as st
import os
import tempfile
import subprocess
import base64
import io
import re
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

# Initialize a list to store image data and messages
if 'image_data_list' not in st.session_state:
    st.session_state.image_data_list = []

print("Environment variables are loaded:", load_dotenv())
api_key = os.getenv("API_KEY")
openai.api_key = os.getenv("API_KEY")

#Connect SQLDatabase
db_path = "data/sqldb"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

#Load LLM
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.0
)

VISUAL_PROMPT = """
You are a skilled data analyst with a strong understanding of data visualization techniques. 

You will always get dataset as input from user. Your task is to analyze the given dataset and suggest the most suitable visualization to highlight the key insights.

Consider the following steps before sending the response.
1. A concise explanation of the recommended visualization technique.
2. **Python code to Create DataFrame from the given dataset** 
3. **Within the python code, Include the required libraries like Matplotlib or Plotly; Apply the recommended visualization chart on the given dataset; and save the output chart as a PNG image with default file name 'output_image.png'**
4. **Make sure the generated python code can be executed without any additional changes.**
5. Give the complete Python code to the user, ensuring correct syntax and indentation.

Consider the following factors:
- Data Types: Analyze the data types of the columns in given dataset to determine appropriate visualizations.
- Data Distribution: Consider the distribution of the data in given dataset to choose suitable visualizations.
- Relationships: Identify relationships between variables in given dataset and select visualizations that highlight these relationships.
- Clarity and Conciseness: Ensure the visualization is clear, concise, and easy to interpret.

**If the given dataset is empty or has only one column or simple one, no need to generate any visualization charts. Just respond to the user politely visualization not required.

**Response Format:**

Data Visualization: {recommended visualization technique}

Python Code for Visualization: {python_code}

""" 

def get_visualization_python_response(prompt):
    messages = [
        {"role": "system", "content": VISUAL_PROMPT}
    ]

    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message.content

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
        subprocess.run(['python', temp_file.name])
        image_path = 'output_image.png'
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
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

# Create a copy of the image data list to avoid modifying the original
image_data_list_copy = st.session_state.image_data_list.copy()

#Display chat message from history on app rerun
for line in st.session_state.conversation_history:
	print(line)
	if line["role"] == "user":
		with st.chat_message("user"):
			st.markdown(line["content"])
	elif line["role"] == "assistant" and "Output Image" not in line["content"]:
		with st.chat_message("assistant"):
			st.markdown(line["content"])
	elif line["role"] == "assistant" and "Output Image" in line["content"] and image_data_list_copy:
		with st.chat_message("assistant"):
			message, image_data = image_data_list_copy.pop(0)
			st.markdown("Here's the visualization:")
			st.image(io.BytesIO(base64.b64decode(image_data)))

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
			
			if query_result:
				visualization_python_code = get_visualization_python_response(query_result)
					
				if "import" in visualization_python_code:
					visualization_code = extract_python_code(visualization_python_code)
					visualization_output = run_python_code_and_return_image(visualization_code)
					
					if visualization_output:
						# Store the image data and a message in the list
						st.session_state.image_data_list.append(("Here's the visualization:", visualization_output))
						st.session_state.conversation_history.append({"role": "assistant", "content": f"AI: Output Image {len(st.session_state.image_data_list)}"})
						st.markdown("Here's the visualization:")
						st.image(io.BytesIO(base64.b64decode(visualization_output)))
						