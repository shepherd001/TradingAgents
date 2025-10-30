import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import json
from langchain_openai import ChatOpenAI
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.interface import route_to_vendor
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = typer.Typer()
console = Console()

def parse_natural_language_command(command: str, llm):
    """
    Uses an LLM to parse a natural language command and extract intent and entities.
    """
    prompt = f"""
    You are a helpful assistant that parses natural language commands for a trading agent.
    Extract the user's intent and any relevant entities from the following command.
    The possible intents are: 'analyze_stock', 'question_answering', 'unknown'.
    The possible entities are: 'ticker', 'date', 'query'.
    The date should be in 'YYYY-MM-DD' format.

    Command: '{command}'

    Return the output as a JSON object with 'intent' and 'entities' as keys.
    For example:
    {{
        "intent": "analyze_stock",
        "entities": {{
            "ticker": "NVDA",
            "date": "2024-05-10"
        }}
    }}
    Or for a question:
    {{
        "intent": "question_answering",
        "entities": {{
            "ticker": "META",
            "query": "why did the stock price raise"
        }}
    }}
    """
    
    response = llm.invoke(prompt)
    
    try:
        # The response from the LLM might be in a markdown code block
        json_str = response.content.strip().replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(json_str)
        return parsed_response
    except (json.JSONDecodeError, KeyError) as e:
        console.print(Panel(f"[bold red]Error parsing LLM response:[/bold red] {e}", title="Error"))
        return {
            "intent": "unknown",
            "entities": {}
        }

def get_summary_from_llm(command: str, news: str, llm):
    """
    Uses an LLM to summarize news and answer a user's question.
    """
    prompt = f"""
    Based on the following news articles, please provide a concise summary that answers the user's question.

    User's Question: "{command}"

    News Articles:
    ---
    {news}
    ---

    Summary:
    """
    
    response = llm.invoke(prompt)
    return response.content

@app.command()
def chat():
    """
    Starts an interactive chat session with the Trading Agent.
    """
    console.print(Panel("[bold green]Welcome to the Natural Language Trading Agent CLI[/bold green]", title="Welcome"))
    console.print("You can type commands like 'analyze NVDA for 2024-05-10' or 'quit' to exit.")

    # Initialize the LLM for parsing commands
    # Make sure you have OPENAI_API_KEY set in your environment
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception as e:
        console.print(Panel(f"[bold red]Error initializing the language model:[/bold red] {e}", title="Error"))
        console.print("Please make sure your OPENAI_API_KEY is set correctly.")
        raise typer.Exit()

    while True:
        command = Prompt.ask("[bold cyan]>>> [/bold cyan]")

        if command.lower() in ["quit", "exit"]:
            break

        if not command.strip():
            continue

        parsed_command = parse_natural_language_command(command, llm)
        intent = parsed_command.get("intent")
        entities = parsed_command.get("entities", {})

        if intent == "analyze_stock":
            ticker = entities.get("ticker")
            date = entities.get("date")

            if not ticker or not date:
                console.print("[bold red]Could not extract ticker or date from the command. Please be more specific.[/bold red]")
                continue

            console.print(f"Analyzing [bold yellow]{ticker}[/bold yellow] for [bold yellow]{date}[/bold yellow]...")

            try:
                # Initialize and run the TradingAgentsGraph
                ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG)
                final_state, decision = ta.propagate(ticker, date)

                # Display the results
                console.print(Panel(f"[bold green]Final Trading Decision:[/bold green] {decision}", title="Result"))
                
                if final_state:
                    # You can add more details from the final_state here if you want
                    summary = (
                        f"Market Report: {'Provided' if final_state.get('market_report') else 'Not Provided'}\n"
                        f"Sentiment Report: {'Provided' if final_state.get('sentiment_report') else 'Not Provided'}\n"
                        f"News Report: {'Provided' if final_state.get('news_report') else 'Not Provided'}\n"
                        f"Fundamentals Report: {'Provided' if final_state.get('fundamentals_report') else 'Not Provided'}"
                    )
                    console.print(Panel(summary, title="Analysis Summary"))

            except Exception as e:
                console.print(Panel(f"[bold red]An error occurred during analysis:[/bold red] {e}", title="Error"))

        elif intent == "question_answering":
            ticker = entities.get("ticker")
            if not ticker:
                console.print("[bold red]Could not extract a ticker symbol from your question. Please be more specific.[/bold red]")
                continue

            console.print(f"Searching for news related to [bold yellow]{ticker}[/bold yellow] and generating a summary...")
            try:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
                
                news = route_to_vendor("get_news", ticker, start_date, end_date)
                
                if news:
                    summary = get_summary_from_llm(command, news, llm)
                    console.print(Panel(summary, title=f"Summary for {ticker}"))
                else:
                    console.print(f"No recent news found for {ticker}.")

            except Exception as e:
                console.print(Panel(f"[bold red]An error occurred while fetching or summarizing news:[/bold red] {e}", title="Error"))
        
        else:
            console.print("[bold red]Sorry, I didn't understand that command. Please try again.[/bold red]")

if __name__ == "__main__":
    app()