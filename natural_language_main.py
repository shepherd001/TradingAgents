import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langchain_openai import ChatOpenAI
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.interface import route_to_vendor
import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()
console = Console()

@tool
def analyze_stock(ticker: str, date: str = None) -> str:
    """
    Analyzes a stock for a given date. If no date is provided, today's date is used.
    Returns a summary of the analysis.
    """
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    console.print(f"Analyzing [bold yellow]{ticker}[/bold yellow] for [bold yellow]{date}[/bold yellow]...")
    try:
        ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG)
        final_state, decision = ta.propagate(ticker, date)
        
        summary = f"Final Trading Decision: {decision}\n"
        if final_state:
            summary += (
                f"Market Report: {'Provided' if final_state.get('market_report') else 'Not Provided'}\n"
                f"Sentiment Report: {'Provided' if final_state.get('sentiment_report') else 'Not Provided'}\n"
                f"News Report: {'Provided' if final_state.get('news_report') else 'Not Provided'}\n"
                f"Fundamentals Report: {'Provided' if final_state.get('fundamentals_report') else 'Not Provided'}"
            )
        return summary
    except Exception as e:
        return f"An error occurred during analysis: {e}"

@tool
def get_news(ticker: str, start_date: str = None, end_date: str = None) -> str:
    """
    Gets news for a stock between a start and end date. If no dates are provided, it will fetch news for the last 3 days.
    Returns a summary of the news.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        
    console.print(f"Searching for news related to [bold yellow]{ticker}[/bold yellow]...")
    try:
        news = route_to_vendor("get_news", ticker, start_date, end_date)
        if news:
            # For simplicity, we are returning the raw news here. 
            # You could use another LLM call to summarize it.
            return news
        else:
            return f"No recent news found for {ticker}."
    except Exception as e:
        return f"An error occurred while fetching news: {e}"

@app.command()
def chat():
    """
    Starts an interactive chat session with the Trading Agent.
    """
    console.print(Panel("[bold green]Welcome to the Natural Language Trading Agent CLI[/bold green]", title="Welcome"))
    console.print("You can type commands like 'analyze NVDA for today' or 'what is the news for TSLA' or 'quit' to exit.")

    try:
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    except Exception as e:
        console.print(Panel(f"[bold red]Error initializing the language model:[/bold red] {e}", title="Error"))
        console.print("Please make sure your OPENAI_API_KEY is set correctly.")
        raise typer.Exit()

    tools = [analyze_stock, get_news]

    system_prompt_content = """You are an expert trading assistant. Your goal is to provide insightful and accurate information about the stock market. You should be conversational and helpful.

You have access to the following tools:

- `analyze_stock`: Use this tool when a user wants to analyze a stock. It can provide a detailed analysis of a stock for a specific date.
- `get_news`: Use this tool when a user asks for the latest news about a stock. It can retrieve recent news articles.

When you receive a query, first think about what the user is asking. Then, decide which tool to use to answer their question. If you need more information, you can ask the user clarifying questions.

Always be friendly and professional in your responses."""

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt_content,
    )

    while True:
        command = Prompt.ask("[bold cyan]>>> [/bold cyan]")

        if command.lower() in ["quit", "exit"]:
            break

        if not command.strip():
            continue

        try:
            response = agent.invoke({"messages": [{"role": "user", "content": command}]})
            final_answer = response['messages'][-1].content
            console.print(Panel(final_answer, title="Agent Response"))
        except Exception as e:
            console.print(Panel(f"[bold red]An error occurred:[/bold red] {e}", title="Error"))

if __name__ == "__main__":
    app()
