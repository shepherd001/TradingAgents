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


def parse_natural_language_command_v1(command: str, llm):

    prompt = f"""
    You are a highly specialized **Stock Market Analyst and Trading Assistant**. Your primary role is to interpret and structure natural language requests for a multi-agent system that performs complex financial analysis and provides actionable trading insights.

    Your task is to:
    1.  **Determine the user's core intent** from the list of specialized trading actions.
    2.  **Extract all relevant entities** required to fulfill the intent.

    ### 🎯 Possible Intents (Trader Actions):
    * **`market_overview`**: For general queries about the current state of the overall stock market (e.g., "How's the market doing today?", "What are the major indices?").
    * **`analyze_stock`**: For requests requiring a technical or fundamental analysis of a single stock or a small list of stocks (e.g., "Analyze Apple's performance," "Show me the 50-day moving average for GOOGL").
    * **`suggest_trade`**: For requests asking for a specific **buy, sell, or hold recommendation** (e.g., "Should I buy Tesla?", "What's the best time to sell Netflix?").
    * **`get_news`**: For requests to retrieve recent, specific news, or events related to a company or the market (e.g., "What are the latest earnings reports for AMD?").
    * **`compare_stocks`**: For requests to compare metrics, performance, or news between two or more specific stocks (e.g., "Compare the P/E ratios of MSFT and ORCL").
    * **`unknown`**: If the command does not clearly match any other intent, or the intent has nothing to do with Stock Market.

    ### 📝 Possible Entities:
    * **`ticker`**: The stock symbol (e.g., 'NVDA', 'MSFT', 'TSLA'). If the user provides a common company name (e.g., "Nvidia"), you must convert it to its standard ticker symbol (e.g., "NVDA") in the output. Can be a single symbol or a list.
    * **`date_range`**: Specifies the period for analysis. **You must convert all relative or descriptive time phrases into a concrete date range object with a `start_date` and `end_date` in 'YYYY-MM-DD' format.** If only one date is mentioned, use it for both `start_date` and `end_date`. Use today's date for 'today', or the most logical recent date for phrases like 'last week' or 'YTD'. If no time is specified, omit this entity.
    * **`metric`**: A specific financial or technical indicator requested (e.g., 'P/E ratio', 'moving average', 'volume', 'resistance level').
    * **`query`**: The specific non-structured text/question detail if the intent is `get_news`.

    ### 💾 Output Format:
    Return the output as a single JSON object with 'intent' and 'entities' as keys. Omit any entity that is not present or relevant to the command.
    **Your response must be a single, valid JSON object and nothing else.**

    **Command:** '{command}'

    **Examples (Note: Assume today's date is 2025-11-06 for the following conversions):**

    1.  **Command:** "analyze NVDA for today"
        ```json
        {{
            "intent": "analyze_stock",
            "entities": {{
                "ticker": "NVDA",
                "date_range": {{
                    "start_date": "2025-11-06",
                    "end_date": "2025-11-06"
                }},
                "query": "analyze NVDA for today"
            }}
        }}
        ```
    2.  **Command:** "Show me Tesla's volume from January 1st to March 31st, 2025."
        ```json
        {{
            "intent": "analyze_stock",
            "entities": {{
                "ticker": "TSLA",
                "date_range": {{
                    "start_date": "2025-01-01",
                    "end_date": "2025-03-31"
                }},
                "metric": "volume"
            }}
        }}
        ```
    3.  **Command:** "Give me a quick rundown on the S&P 500 for the last week."
        ```json
        {{
            "intent": "market_overview",
            "entities": {{
                "ticker": "S&P 500",
                "date_range": {{
                    "start_date": "2025-10-30",
                    "end_date": "2025-11-06"
                }}
            }}
        }}
        ```
    4.  **Command:** "What's the latest news on Tesla's battery day event?"
        ```json
        {{
            "intent": "get_news",
            "entities": {{
                "ticker": "TSLA",
                "query": "battery day event"
            }}
        }}
        ```
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
            "date": "2025-05-10"
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
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
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

        parsed_command = parse_natural_language_command_v1(command, llm)
        intent = parsed_command.get("intent")
        entities = parsed_command.get("entities", {})

        print(parsed_command)

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