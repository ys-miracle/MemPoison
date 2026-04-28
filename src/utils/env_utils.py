from dotenv import load_dotenv
import os

load_dotenv(override= True)

API_KEY = os.getenv("API_KEY")
Base_URL = os.getenv("Base_URL")


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"