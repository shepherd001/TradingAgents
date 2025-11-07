# Import necessary libraries from LangChain and other packages
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file (e.g., for OPENAI_API_KEY)
load_dotenv()

# 1. Initialize the Language Model (LLM)
model = ChatOpenAI(model="gpt-5-mini", temperature=0)

# 2. Define the tools the agent can use
# The @tool decorator makes it easy to convert a function into a tool.
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 3. Create the agent
# The create_agent function is a high-level utility that simplifies agent creation.
# It takes the LLM, the tools, and a system prompt as input.
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# 4. Run the agent
# The agent is invoked with a dictionary containing the user's message.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# 5. Print the final response
# The response is a dictionary containing the full conversation history.
# We are interested in the content of the last message, which is the agent's final answer.
final_answer = response['messages'][-1].content
print(final_answer)