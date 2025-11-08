# Import necessary libraries from LangChain and other packages
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from a .env file (e.g., for OPENAI_API_KEY)
load_dotenv()

# 1. Initialize the Language Model (LLM)
model = ChatOpenAI(model="gpt-5-mini", temperature=0)


client = MultiServerMCPClient(  
    {
        "alphavantage": {
            "transport": "streamable_http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "https://mcp.alphavantage.co/mcp?apikey=HL62T57FQLWUEG7J",
        }
    }
)

async def main():
    mcp_tools = await client.get_tools()

    # 3. Create the agent
    # The create_agent function is a high-level utility that simplifies agent creation.
    # It takes the LLM, the tools, and a system prompt as input.
    agent = create_agent(
        model=model,
        tools=mcp_tools,
        system_prompt="You are Alex, a friendly and expert trading assistant. Your goal is to provide insightful and accurate information about the stock market. You should be conversational and helpful.",
    )

    # 4. Run the agent
    # The agent is invoked with a dictionary containing the user's message.
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "How's the S&P 500 today, let's look at the ETF of SPY, I would like to learn more about current status."}]}
    )

    # 5. Print the final response
    # The response is a dictionary containing the full conversation history.
    # We are interested in the content of the last message, which is the agent's final answer.
    final_answer = response['messages'][-1].content
    print(final_answer)

if __name__ == "__main__":
    asyncio.run(main())