import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS

# Load environment variables from .env
load_dotenv()

# --- 1. Define Custom Tools ---

@tool
def search_web(query: str) -> str:
    """Search the web for real-time information, news, facts, or data using DuckDuckGo."""
    print(f"\n[Tool Execution] Searching the web for: '{query}'")
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No relevant search results found."
            formatted_results = "\n".join([f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results])
            return formatted_results
    except Exception as e:
        return f"Error executing web search: {str(e)}"

@tool
def calculate_math(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid mathematical string like '1500 * 0.08' or '(450 + 120) / 3'."""
    print(f"\n[Tool Execution] Evaluating math expression: {expression}")
    try:
        # Using a restricted eval for basic math safety
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Use only numbers and basic operators (+, -, *, /, parentheses)."
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """Read the contents of a local file from the workspace (e.g., './data/company_policy.txt')."""
    print(f"\n[Tool Execution] Reading local file: {file_path}")
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def run_autonomous_agent():
    # 2. Gather tools into a list
    tools = [search_web, calculate_math, read_local_file]

    # 3. Initialize Gemini LLM (Supports tool calling natively)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    # 4. Create the Agent Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an advanced autonomous AI assistant. You have access to tools to search the web, "
            "perform calculations, and read local workspace files.\n"
            "Analyze the user's request, choose the appropriate tool(s) when necessary, observe the results, "
            "and provide a comprehensive final answer."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 5. Construct the Tool-Calling Agent and Executor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,  # Enables printing the full reasoning chain (Thought -> Action -> Observation)
        handle_parsing_errors=True
    )

    # 6. Test Compound Research Queries
    query = (
        "First, read the local file './data/company_policy.txt' to find the professional development stipend amount. "
        "Then, use the web search tool to find the current stock price or market status of Google (Alphabet Inc.), "
        "and calculate what percentage of the stipend that stock price represents."
    )

    print(f"--- Running Autonomous Agent ---\nUser Prompt:\n{query}\n")
    
    response = agent_executor.invoke({"input": query})
    
    print("\n[Final Answer]:")
    print(response["output"])

if __name__ == "__main__":
    run_autonomous_agent()