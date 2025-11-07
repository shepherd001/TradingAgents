
import os
from langchain_openai import ChatOpenAI
from natural_language_main import parse_natural_language_command_v1
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_test():
    """
    Tests the parse_natural_language_command_v1 function.
    """
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    except Exception as e:
        print(f"Error initializing the language model: {e}")
        return

    command = "What's the latest news on the Tesla general meeting today"
    parsed_command = parse_natural_language_command_v1(command, llm)
    print(parsed_command)

if __name__ == "__main__":
    run_test()
