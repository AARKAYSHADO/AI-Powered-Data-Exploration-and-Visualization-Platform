import os
import requests

from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Load environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Debugging prints to verify values
print("API Key:", api_key)
print("Endpoint:", endpoint)
