import os
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

# Load environment variables
load_dotenv()  # This loads the .env file
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Check if environment variables are loaded correctly
if not api_key or not endpoint:
    raise ValueError("Environment variables AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT are not set.")

# Set up LangChain with AzureOpenAI
deployment_name = "gpt-4"  # Use your deployment name for gpt-4

llm = AzureOpenAI(
    deployment_name=deployment_name,
    openai_api_key=api_key,
    openai_api_base=endpoint,
    temperature=0.3
)

# Define the few-shot prompt template for SQL query generation
prompt_template = """
Convert the following business user queries into SQL code:

Business Query: Find the names and ages of all employees older than 30.
SQL Code:
SELECT name, age
FROM employees
WHERE age > 30;

Business Query: Get the total sales for each product.
SQL Code:
SELECT product_id, SUM(sales) as total_sales
FROM sales_data
GROUP BY product_id;

Business Query: List all customers who made a purchase in the last 30 days.
SQL Code:
SELECT customer_id, customer_name
FROM customers
WHERE purchase_date >= DATE('now', '-30 days');

Business Query: Show the average salary by department.
SQL Code:
SELECT department, AVG(salary) as average_salary
FROM employees
GROUP BY department;

---

Business Query: {business_query}
SQL Code:
"""

# Function to generate SQL query using LangChain
def generate_sql_query(business_query):
    # Create a LangChain prompt template
    prompt = PromptTemplate(input_variables=["business_query"], template=prompt_template)
    
    # Create a LangChain LLMChain with the prompt and the model (AzureOpenAI)
    chain = LLMChain(prompt=prompt, llm=llm)
    
    # Generate SQL query from the model
    sql_query = chain.run(business_query)
    
    return sql_query

# Example usage
business_query = "Find the total number of orders placed in the last week."
sql_query = generate_sql_query(business_query)
print("Generated SQL Query:", sql_query)
