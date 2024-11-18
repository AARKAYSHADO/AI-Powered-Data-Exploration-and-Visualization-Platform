# !pip install langchain_openai langchain_community langchain pymysql chromadb db-sqlite3 -q
# database
import sqlite3
import pandas as pd
import os
from pprint import pprint
import sqlalchemy as sa
from langchain_community.utilities import SQLDatabase

# table selector
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# History
from langchain.memory import ChatMessageHistory

db = SQLDatabase.from_uri("sqlite:////content/drive/MyDrive/Colab Notebooks/datasets/chinook.db")


print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)

history = ChatMessageHistory()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generate_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

query = generate_query.invoke({"question": "how many employees are there`"})
execute_query.invoke(query)


# prompt


table_details = """
Table Name: employees
Table Description: employees table stores employee data such as id, last name, first name, etc. It also has a field named ReportsTo to specify who reports to whom.


Table Name: customers
Table Description: customers table stores customer data.


Table Name: invoices
Table Description: The invoices table stores invoice header data 


Table Name: invoice_items
Table Description: invoice_items table stores the invoice line items data.


Table Name: artists
Table Description: artists table stores artist data. It is a simple table that contains the id and name.


Table Name: albums
Table Description: albums table stores data about a list of tracks. Each album belongs to one artist, but an artist may have multiple 

albums.
Table Name: media_types
Table Description: media_types table stores media types such as MPEG audio and AAC audio files.


Table Name: genres
Table Description: genres table stores music types such as rock, jazz, metal, etc.


Table Name: tracks
Table Description: tracks table stores the data of songs. Each track belongs to one album.


Table Name: playlists
Table Description: playlists table stores data about playlists. Each playlist contains a list of tracks.Each track may belong to 
multiple playlists. The relationship between the playlists and tracks tables is many-to-many.


Table Name: playlist_track
Table Description: The playlist_track table is used to reflect playlists relationship.
"""

table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)

def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables

select_table = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables

few_shot_examples =  [
     {
         "input": "List all customers in France with a credit limit over 20,000.",
         "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
     },
     {
         "input": "Get the highest payment amount made by any customer.",
         "query": "SELECT MAX(amount) FROM payments;"
     },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

# vectoize the examples 
vectorstore = Chroma()
vectorstore.delete_collection()
dynamic_example_selector = SemanticSimilarityExampleSelector.from_examples(
     examples,
     OpenAIEmbeddings(),
     vectorstore,
     k=1,
     input_keys=["input"],
)

#  example_selector.select_examples({"input": "how many employees we have?"})

# Dynamically select the examples 
dynamic_few_shot_prompt = FewShotChatMessagePromptTemplate(
     example_prompt=example_prompt,
     example_selector=dynamic_example_selector,
     input_variables=["input","top_k"],
 )
#  print(few_shot_prompt.format(input="How many products are there?"))

prompt = ChatPromptTemplate.from_messages(
  [
      ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries. Those examples are just for referecne and hsould be considered while answering follow up questions"),
      dynamic_few_shot_prompt,
      MessagesPlaceholder(variable_name="messages"),
      ("human", "{input}"),
  ]
)

generate_query = create_sql_query_chain(llm, db, prompt)

rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(table_names_to_use=select_table) |
    RunnablePassthrough.assign(query=generate_query)
    .assign(result=itemgetter("query") | execute_query)
    | rephrase_answer
)

question = "how many employees are there"
chain.invoke({"question": f"{question}", "table_info":"some table info", "messages":history.messages})


# iterate

while True : 
  question = input("Enter your Question : ")
  if question == '-1':
    break
  response = chain.invoke({"question": f"{question}", "table_info":"some table info", "messages":history.messages})
  print(response)
  history.add_user_message(question)
  history.add_ai_message(response)