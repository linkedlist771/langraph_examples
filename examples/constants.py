from dotenv import load_dotenv
from loguru import logger
from  os import environ
load_dotenv(override=True)

MODEL = environ.get("OPENAI_MODEL")
BASE_URL = environ.get("OPENAI_BASE_URL")
API_KEY = environ.get("OPENAI_API_KEY")