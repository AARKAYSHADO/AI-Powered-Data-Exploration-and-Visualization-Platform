import os
import requests
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Check if environment variables are loaded correctly
if not api_key or not endpoint:
    raise ValueError("Environment variables AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT are not set.")

# Define the model deployment name - continue using gpt-4
deployment_name = "gpt-4"  # Keep gpt-4

# Set up the headers for the API request
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

# Define the API URL for chat completions
url = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-05-15"

# Define a few-shot prompt for SQL query generation
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

Business Query: {}
SQL Code:
"""

# Function to generate SQL query using Azure OpenAI with gpt-4
def generate_sql_query(business_query):
    # Format the prompt with the business query
    prompt = prompt_template.format(business_query)
    
    # Define the request payload for chat-based completion (gpt-4)
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that converts business queries into SQL."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 150,
        "top_p": 0.9,
        "stop": ["---"]
    }
    
    # Make the API request to Azure OpenAI
    response = requests.post(url, headers=headers, json=data)
    
    # Check for successful response
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        print("Failed to connect to Azure OpenAI:", response.status_code, response.text)
        return None

# Example usage
business_query = "Find the total number of orders placed in the last week."
sql_query = generate_sql_query(business_query)
print("Generated SQL Query:", sql_query)
