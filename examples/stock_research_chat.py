#!/usr/bin/env python3
"""
Conversational Stock Research Assistant using Agora + Alpha Vantage + OpenAI

A chat-based interface where you can discuss stocks, find competitors,
compare companies, and research markets interactively.

🔑 PASTE YOUR API KEYS BELOW (lines 22-23)
"""

import os
import sys
import asyncio
import requests
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# ============================================================================
# 🔑 PASTE YOUR API KEYS HERE
# ============================================================================

OPENAI_API_KEY = ""  # ← PASTE YOUR OPENAI KEY HERE
ALPHA_VANTAGE_KEY = ""  # ← PASTE YOUR ALPHA VANTAGE KEY HERE

# ============================================================================

# Validate keys
if not OPENAI_API_KEY or not ALPHA_VANTAGE_KEY:
    print("\n" + "="*70)
    print("❌ API KEYS MISSING!")
    print("="*70)
    print("\n📝 Edit this file and paste your keys on lines 22-23")
    print("\n🔑 Get your keys:")
    print("   • OpenAI: https://platform.openai.com/api-keys")
    print("   • Alpha Vantage (FREE): https://www.alphavantage.co/support/#api-key")
    print("\n" + "="*70 + "\n")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ALPHA_VANTAGE_KEY"] = ALPHA_VANTAGE_KEY

# Check dependencies
try:
    import gradio as gr
    from openai import AsyncOpenAI
    from agora.agora_tracer import init_traceloop, agora_node, TracedAsyncFlow
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install: pip install gradio openai traceloop-sdk opentelemetry-api opentelemetry-sdk")
    sys.exit(1)

# ============================================================================
# INITIALIZE
# ============================================================================

print("🚀 Initializing Stock Research Chat Assistant...")

# Initialize with both console and file export
init_traceloop(
    app_name="stock_chat_assistant",
    export_to_console=True,  # Show in terminal
    export_to_file="telemetry_traces.jsonl",  # Save to file
    disable_content_logging=True
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print("✅ Ready!")
print("📊 Telemetry: Console + telemetry_traces.jsonl\n")

# ============================================================================
# ALPHA VANTAGE API
# ============================================================================

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

def fetch_alpha_vantage(function: str, symbol: str = None, **kwargs) -> Dict[str, Any]:
    """Fetch data from Alpha Vantage API."""
    params = {"function": function, "apikey": ALPHA_VANTAGE_KEY, **kwargs}
    if symbol:
        params["symbol"] = symbol

    try:
        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data:
            return {"error": "API rate limit (5/min). Wait 60 seconds."}

        return data
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# AGORA WORKFLOW NODES
# ============================================================================

@agora_node(name="InterpretQuery")
async def interpret_query(shared: Dict[str, Any]) -> str:
    """
    Use AI to understand what the user is asking for and decide what actions to take.
    """
    user_message = shared["user_message"]
    chat_history = shared.get("chat_history", [])

    # Build context from history
    history_context = "\n".join([
        f"{msg['role']}: {msg['content'][:200]}"
        for msg in chat_history[-4:]  # Last 4 messages
    ]) if chat_history else "No previous context"

    # Ask AI to interpret the query
    system_prompt = """You are a stock research assistant. Analyze the user's question and determine what actions to take.

Available actions:
1. SEARCH_SYMBOL - Search for stock symbols by company name or keyword
2. GET_QUOTE - Get real-time quote for specific symbol(s)
3. GET_OVERVIEW - Get company fundamentals and overview
4. COMPARE_STOCKS - Compare multiple stocks
5. FIND_COMPETITORS - Find companies in same sector/industry
6. GENERAL_CHAT - Answer general questions about markets, investing, etc.

Respond in JSON format:
{
    "action": "ACTION_NAME",
    "symbols": ["AAPL", "MSFT"],  // if applicable
    "search_query": "biotech cancer",  // if searching
    "response_hint": "Brief hint of what to include in response"
}

Examples:
User: "Tell me about Apple" -> {"action": "SEARCH_SYMBOL", "search_query": "Apple"}
User: "Compare AAPL and MSFT" -> {"action": "COMPARE_STOCKS", "symbols": ["AAPL", "MSFT"]}
User: "What are Tesla's competitors?" -> {"action": "FIND_COMPETITORS", "symbols": ["TSLA"]}
User: "Which biotech companies work on cancer drugs?" -> {"action": "SEARCH_SYMBOL", "search_query": "biotech cancer"}
"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{history_context}\n\nCurrent question: {user_message}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        # Parse AI response
        ai_response = response.choices[0].message.content.strip()

        # Try to extract JSON
        if "```json" in ai_response:
            ai_response = ai_response.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_response:
            ai_response = ai_response.split("```")[1].split("```")[0].strip()

        intent = json.loads(ai_response)
        shared["intent"] = intent

        action = intent.get("action", "GENERAL_CHAT")

        # Route based on action
        if action == "SEARCH_SYMBOL":
            return "search_symbol"
        elif action == "GET_QUOTE":
            return "get_quote"
        elif action in ["GET_OVERVIEW", "COMPARE_STOCKS", "FIND_COMPETITORS"]:
            return "get_overview"
        else:
            return "general_chat"

    except Exception as e:
        print(f"Error interpreting query: {e}")
        shared["intent"] = {"action": "GENERAL_CHAT", "error": str(e)}
        return "general_chat"


@agora_node(name="SearchSymbol", max_retries=2, wait=1)
async def search_symbol(shared: Dict[str, Any]) -> str:
    """Search for stock symbols by keyword."""
    intent = shared.get("intent", {})
    search_query = intent.get("search_query", "")

    if not search_query:
        shared["search_results"] = []
        return "get_overview"

    data = await asyncio.to_thread(
        fetch_alpha_vantage,
        "SYMBOL_SEARCH",
        keywords=search_query
    )

    if "error" in data:
        shared["search_results"] = []
        shared["error_message"] = data["error"]
        return "general_chat"

    matches = data.get("bestMatches", [])[:5]  # Top 5 results

    symbols = []
    for match in matches:
        symbol = match.get("1. symbol", "")
        name = match.get("2. name", "")
        region = match.get("4. region", "")
        symbols.append({"symbol": symbol, "name": name, "region": region})

    shared["search_results"] = symbols
    shared["symbols_to_fetch"] = [s["symbol"] for s in symbols if s.get("region") == "United States"][:3]

    return "get_overview" if shared["symbols_to_fetch"] else "general_chat"


@agora_node(name="GetQuote", max_retries=2, wait=1)
async def get_quote(shared: Dict[str, Any]) -> str:
    """Fetch real-time quotes for symbols."""
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))

    if not symbols:
        return "general_chat"

    quotes = []
    for symbol in symbols[:5]:  # Limit to 5 to avoid rate limits
        data = await asyncio.to_thread(
            fetch_alpha_vantage,
            "GLOBAL_QUOTE",
            symbol=symbol
        )

        if "error" not in data:
            quote = data.get("Global Quote", {})
            if quote:
                quotes.append({
                    "symbol": symbol,
                    "price": quote.get("05. price", "N/A"),
                    "change": quote.get("09. change", "N/A"),
                    "change_percent": quote.get("10. change percent", "N/A"),
                    "volume": quote.get("06. volume", "N/A")
                })

        # Small delay to avoid rate limits
        await asyncio.sleep(0.3)

    shared["quotes"] = quotes
    return "get_overview"


@agora_node(name="GetOverview", max_retries=2, wait=1)
async def get_overview(shared: Dict[str, Any]) -> str:
    """Fetch company overviews."""
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))

    if not symbols:
        return "generate_response"

    overviews = []
    for symbol in symbols[:3]:  # Limit to 3 for detailed info
        data = await asyncio.to_thread(
            fetch_alpha_vantage,
            "OVERVIEW",
            symbol=symbol
        )

        if "error" not in data and data:
            overviews.append({
                "symbol": symbol,
                "name": data.get("Name", "N/A"),
                "sector": data.get("Sector", "N/A"),
                "industry": data.get("Industry", "N/A"),
                "market_cap": data.get("MarketCapitalization", "N/A"),
                "pe_ratio": data.get("PERatio", "N/A"),
                "description": data.get("Description", "N/A")[:400]
            })

        await asyncio.sleep(0.3)

    shared["overviews"] = overviews
    return "generate_response"


@agora_node(name="GeneralChat")
async def general_chat(shared: Dict[str, Any]) -> str:
    """Handle general questions without API calls."""
    shared["response_type"] = "general"
    return "generate_response"


@agora_node(name="GenerateResponse", max_retries=2, wait=1)
async def generate_response(shared: Dict[str, Any]) -> str:
    """Generate natural language response based on collected data."""
    user_message = shared["user_message"]
    intent = shared.get("intent", {})

    # Collect all data
    search_results = shared.get("search_results", [])
    quotes = shared.get("quotes", [])
    overviews = shared.get("overviews", [])
    error_message = shared.get("error_message", "")

    # Build context for AI
    context = f"User asked: {user_message}\n\n"

    if error_message:
        context += f"Error: {error_message}\n\n"

    if search_results:
        context += "Search Results:\n"
        for s in search_results:
            context += f"- {s['symbol']}: {s['name']} ({s.get('region', 'N/A')})\n"
        context += "\n"

    if quotes:
        context += "Real-time Quotes:\n"
        for q in quotes:
            context += f"- {q['symbol']}: ${q['price']} ({q['change_percent']})\n"
        context += "\n"

    if overviews:
        context += "Company Details:\n"
        for o in overviews:
            context += f"\n{o['symbol']} - {o['name']}:\n"
            context += f"  Sector: {o['sector']} | Industry: {o['industry']}\n"
            context += f"  Market Cap: ${o['market_cap']} | P/E: {o['pe_ratio']}\n"
            context += f"  {o['description'][:200]}...\n"

    # Generate response
    system_prompt = """You are a helpful stock research assistant. Based on the data provided, give a clear, conversational response.

Guidelines:
- Be conversational and friendly
- Highlight key insights
- If comparing companies, point out differences
- If user asks about competitors, suggest related companies
- Keep responses concise but informative
- Don't give financial advice, just present data
- If no data available, explain why and suggest alternatives
"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=600
        )

        bot_response = response.choices[0].message.content
        shared["bot_response"] = bot_response

    except Exception as e:
        shared["bot_response"] = f"I encountered an error generating the response: {str(e)}"

    return "complete"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_chat_flow() -> TracedAsyncFlow:
    """Build the conversational workflow."""
    flow = TracedAsyncFlow("StockChatFlow")

    flow.start(interpret_query)

    # Routing based on intent
    interpret_query - "search_symbol" >> search_symbol
    interpret_query - "get_quote" >> get_quote
    interpret_query - "get_overview" >> get_overview
    interpret_query - "general_chat" >> general_chat

    # All paths lead to response generation
    search_symbol - "get_overview" >> get_overview
    search_symbol - "general_chat" >> general_chat

    get_quote - "get_overview" >> get_overview

    get_overview - "generate_response" >> generate_response
    general_chat - "generate_response" >> generate_response

    return flow


# ============================================================================
# GRADIO CHAT INTERFACE
# ============================================================================

# Store conversation state
conversation_state = {
    "chat_history": []
}

async def chat_response(message: str, history: List[Tuple[str, str]]):
    """Handle chat messages."""
    if not message.strip():
        return history

    # Build chat history for context
    chat_history = []
    for user_msg, bot_msg in history:
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": bot_msg})

    # Create shared state
    shared = {
        "user_message": message,
        "chat_history": chat_history,
        "start_time": datetime.now()
    }

    # Run workflow
    flow = build_chat_flow()

    try:
        import time
        start = time.time()
        await flow.run_async(shared)
        elapsed = time.time() - start

        response = shared.get("bot_response", "I'm not sure how to respond to that.")

        # Add execution stats
        stats = f"\n\n---\n*⏱️ Workflow executed in {elapsed:.2f}s*"

        # Add node execution info if available
        if shared.get("search_results"):
            stats += f" | 🔍 Found {len(shared['search_results'])} companies"
        if shared.get("quotes"):
            stats += f" | 📊 Fetched {len(shared['quotes'])} quotes"
        if shared.get("overviews"):
            stats += f" | 📋 Got {len(shared['overviews'])} overviews"

        response += stats

    except Exception as e:
        response = f"Sorry, I encountered an error: {str(e)}"

    # Update history
    history.append((message, response))

    return history


def view_telemetry():
    """View telemetry data from file."""
    try:
        with open("telemetry_traces.jsonl", "r") as f:
            lines = f.readlines()[-20:]  # Last 20 traces

        traces = []
        for line in lines:
            try:
                trace = json.loads(line)
                name = trace.get("name", "unknown")
                duration = trace.get("attributes", {}).get("duration_ms", "N/A")
                traces.append(f"{name}: {duration}ms")
            except:
                continue

        return "\n".join(traces) if traces else "No telemetry data yet"
    except FileNotFoundError:
        return "No telemetry file found. Make a query first!"
    except Exception as e:
        return f"Error reading telemetry: {str(e)}"


def create_chat_interface():
    """Create Gradio chat interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Stock Research Chat") as demo:
        gr.Markdown("""
        # 💬 Stock Research Chat Assistant
        ### Powered by Agora + Alpha Vantage + OpenAI

        Ask me anything about stocks! Examples:
        - "Tell me about Apple and Microsoft"
        - "Which biotech companies work on cancer treatments?"
        - "Compare TSLA and F"
        - "What are Amazon's competitors?"
        - "Show me chip manufacturers"

        *I'll search for companies, fetch data, and answer your questions!*
        """)

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                label="Stock Research Assistant",
                height=500,
                show_copy_button=True,
                type="tuples"
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about stocks, companies, or markets...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("""
            ---
            **Tips:**
            - Ask follow-up questions to dive deeper
            - Request comparisons between companies
            - Search by company name or industry
            - Ask about competitors in a sector

            **Rate Limits:** Alpha Vantage free tier = 5 API calls/minute
            """)

            # Handle submission
            async def submit_message(message, history):
                return await chat_response(message, history)

            msg.submit(submit_message, [msg, chatbot], [chatbot])
            submit.click(submit_message, [msg, chatbot], [chatbot])

        with gr.Tab("📊 Telemetry"):
            gr.Markdown("""
            ## 📊 Workflow Telemetry

            View execution traces from Agora nodes. Each query's workflow is tracked with:
            - Node execution times
            - Retry attempts
            - Error tracking
            - API call performance
            """)

            telemetry_output = gr.Textbox(
                label="Recent Traces (Last 20)",
                lines=20,
                max_lines=30,
                interactive=False
            )

            refresh_btn = gr.Button("🔄 Refresh Telemetry", variant="secondary")
            refresh_btn.click(view_telemetry, outputs=telemetry_output)

            gr.Markdown("""
            ---
            **Telemetry Files:**
            - `telemetry_traces.jsonl` - Complete trace log (JSONL format)
            - Console output - Real-time OpenTelemetry spans

            Each trace includes: timestamp, node name, duration, attributes, errors
            """)

    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch the chat interface."""
    print("\n" + "="*60)
    print("💬 Stock Research Chat Assistant")
    print("="*60)
    print("\n🔧 Configuration:")
    print("   • OpenAI API: ✅ Configured")
    print("   • Alpha Vantage: ✅ Configured")
    print("   • Telemetry: ✅ Enabled")
    print("\n🚀 Launching chat interface...\n")

    demo = create_chat_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,  # Different port from stock_research_app.py
        show_error=True
    )


if __name__ == "__main__":
    main()
