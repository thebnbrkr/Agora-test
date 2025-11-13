#!/usr/bin/env python3
"""
Conversational Stock Research Assistant - Full Agora Framework Demo

Features:
- @agora_node decorator for workflow nodes
- TracedAsyncFlow for orchestration
- Persistent memory across conversations
- Parallel API calls for efficiency
- Automatic telemetry with OpenTelemetry
- Visual workflow dashboard

🔑 PASTE YOUR API KEYS BELOW (lines 26-27)
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

if not OPENAI_API_KEY or not ALPHA_VANTAGE_KEY:
    print("\n" + "="*70)
    print("❌ API KEYS MISSING!")
    print("="*70)
    print("\n📝 Edit lines 26-27 and paste your keys")
    print("\n🔑 Get keys:")
    print("   • OpenAI: https://platform.openai.com/api-keys")
    print("   • Alpha Vantage (FREE): https://www.alphavantage.co/support/#api-key")
    print("\n" + "="*70 + "\n")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ALPHA_VANTAGE_KEY"] = ALPHA_VANTAGE_KEY

try:
    import gradio as gr
    from openai import AsyncOpenAI
    from agora.agora_tracer import init_traceloop, agora_node, TracedAsyncFlow, TracedAsyncNode
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("\n📦 Install: pip install gradio openai traceloop-sdk opentelemetry-api opentelemetry-sdk")
    sys.exit(1)

# ============================================================================
# INITIALIZE WITH TELEMETRY
# ============================================================================

print("🚀 Initializing Agora Stock Research Chat...")

init_traceloop(
    app_name="agora_stock_chat_demo",
    export_to_console=True,
    export_to_file="agora_workflow_traces.jsonl",
    disable_content_logging=True
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print("✅ Agora Framework Ready!")
print("📊 Telemetry: Console + agora_workflow_traces.jsonl")
print("💾 Memory: Persistent conversation context\n")

# ============================================================================
# CONVERSATION MEMORY - Agora Feature: State Management
# ============================================================================

class ConversationMemory:
    """
    Persistent memory across workflow executions.
    Demonstrates Agora's state management capabilities.
    """
    def __init__(self):
        self.history = []
        self.mentioned_symbols = set()
        self.last_intent = None
        self.context = {}
        self.workflow_traces = []

    def add_message(self, role: str, content: str):
        """Add message to history."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 10 messages
        self.history = self.history[-10:]

    def add_symbols(self, symbols: List[str]):
        """Track symbols user has asked about."""
        self.mentioned_symbols.update(symbols)

    def get_context_string(self) -> str:
        """Build context string for AI."""
        ctx = []
        if self.mentioned_symbols:
            ctx.append(f"Previously discussed: {', '.join(list(self.mentioned_symbols)[-5:])}")
        if self.last_intent:
            ctx.append(f"Last action: {self.last_intent}")
        return " | ".join(ctx) if ctx else "No previous context"

    def add_workflow_trace(self, trace: Dict[str, Any]):
        """Store workflow execution trace."""
        self.workflow_traces.append(trace)
        # Keep last 20 traces
        self.workflow_traces = self.workflow_traces[-20:]

    def get_last_trace(self) -> Dict[str, Any]:
        """Get most recent workflow trace."""
        return self.workflow_traces[-1] if self.workflow_traces else None


# Global memory instance
memory = ConversationMemory()

# ============================================================================
# ALPHA VANTAGE API
# ============================================================================

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

def fetch_alpha_vantage(function: str, symbol: str = None, **kwargs) -> Dict[str, Any]:
    """Fetch from Alpha Vantage API with error handling."""
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
            return {"error": "Rate limit (5/min). Wait 60s."}

        return data
    except Exception as e:
        return {"error": str(e)}


async def fetch_multiple_quotes_parallel(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch multiple quotes in parallel using asyncio.
    Demonstrates efficient async processing with Agora.
    """
    async def fetch_one(symbol):
        return await asyncio.to_thread(fetch_alpha_vantage, "GLOBAL_QUOTE", symbol=symbol)

    results = await asyncio.gather(*[fetch_one(s) for s in symbols[:5]])

    quotes = []
    for symbol, data in zip(symbols, results):
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
    return quotes

# ============================================================================
# AGORA NODES - Workflow Building Blocks
# ============================================================================

@agora_node(name="InterpretQuery", max_retries=2, wait=1)
async def interpret_query(shared: Dict[str, Any]) -> str:
    """
    AI-powered intent detection with conversation memory.
    Uses Agora's automatic retry and telemetry tracking.
    """
    user_message = shared["user_message"]

    # Use memory for context
    context = memory.get_context_string()
    recent_history = memory.history[-4:] if memory.history else []

    system_prompt = """You are a stock research assistant. Analyze the query and determine actions.

Available actions:
1. SEARCH_SYMBOL - Search for stock symbols by name/keyword
2. GET_QUOTE - Get real-time quotes for specific symbols
3. GET_OVERVIEW - Get company fundamentals
4. COMPARE_STOCKS - Compare multiple stocks
5. FIND_COMPETITORS - Find competitors in sector
6. GENERAL_CHAT - Answer general questions

Respond in JSON:
{
    "action": "ACTION_NAME",
    "symbols": ["AAPL", "MSFT"],
    "search_query": "biotech cancer",
    "response_hint": "Brief hint"
}
"""

    history_context = "\n".join([f"{m['role']}: {m['content'][:150]}" for m in recent_history])

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nHistory:\n{history_context}\n\nQuery: {user_message}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        ai_response = response.choices[0].message.content.strip()

        # Extract JSON
        if "```json" in ai_response:
            ai_response = ai_response.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_response:
            ai_response = ai_response.split("```")[1].split("```")[0].strip()

        intent = json.loads(ai_response)
        shared["intent"] = intent

        # Update memory
        memory.last_intent = intent.get("action")
        if intent.get("symbols"):
            memory.add_symbols(intent["symbols"])

        action = intent.get("action", "GENERAL_CHAT")

        # Route based on action
        if action == "SEARCH_SYMBOL":
            return "search_symbol"
        elif action == "GET_QUOTE":
            return "get_quote_parallel"
        elif action in ["GET_OVERVIEW", "COMPARE_STOCKS", "FIND_COMPETITORS"]:
            return "get_overview"
        else:
            return "general_chat"

    except Exception as e:
        shared["intent"] = {"action": "GENERAL_CHAT", "error": str(e)}
        return "general_chat"


@agora_node(name="SearchSymbol", max_retries=2, wait=1)
async def search_symbol(shared: Dict[str, Any]) -> str:
    """Search for stock symbols with Alpha Vantage."""
    intent = shared.get("intent", {})
    search_query = intent.get("search_query", "")

    if not search_query:
        shared["search_results"] = []
        return "general_chat"

    data = await asyncio.to_thread(
        fetch_alpha_vantage,
        "SYMBOL_SEARCH",
        keywords=search_query
    )

    if "error" in data:
        shared["search_results"] = []
        shared["error_message"] = data["error"]
        return "general_chat"

    matches = data.get("bestMatches", [])[:5]

    symbols = []
    for match in matches:
        symbol = match.get("1. symbol", "")
        name = match.get("2. name", "")
        region = match.get("4. region", "")
        symbols.append({"symbol": symbol, "name": name, "region": region})

    shared["search_results"] = symbols
    shared["symbols_to_fetch"] = [s["symbol"] for s in symbols if s.get("region") == "United States"][:3]

    # Update memory with found symbols
    if shared["symbols_to_fetch"]:
        memory.add_symbols(shared["symbols_to_fetch"])

    return "get_quote_parallel" if shared["symbols_to_fetch"] else "general_chat"


@agora_node(name="GetQuoteParallel", max_retries=2, wait=1)
async def get_quote_parallel(shared: Dict[str, Any]) -> str:
    """
    Fetch multiple quotes in PARALLEL for efficiency.
    Demonstrates Agora's async capabilities.
    """
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))

    if not symbols:
        return "general_chat"

    # Parallel fetch - much faster!
    quotes = await fetch_multiple_quotes_parallel(symbols)
    shared["quotes"] = quotes

    # Update memory
    memory.add_symbols([q["symbol"] for q in quotes])

    return "get_overview"


@agora_node(name="GetOverview", max_retries=2, wait=1)
async def get_overview(shared: Dict[str, Any]) -> str:
    """Fetch company overviews."""
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))

    if not symbols:
        return "generate_response"

    overviews = []
    for symbol in symbols[:3]:
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
                "dividend_yield": data.get("DividendYield", "N/A"),
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
    """
    Generate natural language response with conversation memory.
    Final node in workflow.
    """
    user_message = shared["user_message"]
    intent = shared.get("intent", {})

    # Collect all data
    search_results = shared.get("search_results", [])
    quotes = shared.get("quotes", [])
    overviews = shared.get("overviews", [])
    error_message = shared.get("error_message", "")

    # Build context
    context = f"User asked: {user_message}\n\n"
    context += f"Conversation context: {memory.get_context_string()}\n\n"

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

    system_prompt = """You are a helpful stock research assistant with conversation memory.

Guidelines:
- Be conversational and remember previous context
- Reference past discussions when relevant
- Highlight key insights and differences
- Suggest related companies or follow-up questions
- Keep responses concise but informative
- No financial advice, just present data
"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=700
        )

        bot_response = response.choices[0].message.content
        shared["bot_response"] = bot_response

    except Exception as e:
        shared["bot_response"] = f"Error generating response: {str(e)}"

    return "complete"


# ============================================================================
# WORKFLOW BUILDER - TracedAsyncFlow
# ============================================================================

def build_chat_flow() -> TracedAsyncFlow:
    """
    Build the conversational workflow with automatic telemetry.
    Demonstrates Agora's TracedAsyncFlow for orchestration.
    """
    flow = TracedAsyncFlow("StockChatWorkflow")

    flow.start(interpret_query)

    # Define routing graph
    interpret_query - "search_symbol" >> search_symbol
    interpret_query - "get_quote_parallel" >> get_quote_parallel
    interpret_query - "get_overview" >> get_overview
    interpret_query - "general_chat" >> general_chat

    search_symbol - "get_quote_parallel" >> get_quote_parallel
    search_symbol - "general_chat" >> general_chat

    get_quote_parallel - "get_overview" >> get_overview

    get_overview - "generate_response" >> generate_response
    general_chat - "generate_response" >> generate_response

    return flow


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

async def chat_response(message: str, history: List[Tuple[str, str]]):
    """Handle chat messages with full workflow execution."""
    if not message.strip():
        return history

    # Add to memory
    memory.add_message("user", message)

    # Create shared state
    shared = {
        "user_message": message,
        "start_time": datetime.now(),
        "_execution_log": []
    }

    # Run Agora workflow
    flow = build_chat_flow()

    try:
        import time
        start = time.time()
        await flow.run_async(shared)
        elapsed = time.time() - start

        response = shared.get("bot_response", "I'm not sure how to respond.")

        # Build trace for dashboard
        trace = {
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "total_time": elapsed,
            "intent": shared.get("intent", {}),
            "search_results": shared.get("search_results", []),
            "quotes": shared.get("quotes", []),
            "overviews": shared.get("overviews", []),
            "memory_context": memory.get_context_string()
        }

        memory.add_workflow_trace(trace)
        memory.add_message("assistant", response)

        # Add stats
        stats = f"\n\n---\n*⏱️ {elapsed:.2f}s*"
        if shared.get("search_results"):
            stats += f" | 🔍 {len(shared['search_results'])} found"
        if shared.get("quotes"):
            stats += f" | 📊 {len(shared['quotes'])} quotes (parallel)"
        if shared.get("overviews"):
            stats += f" | 📋 {len(shared['overviews'])} overviews"

        response += stats

    except Exception as e:
        response = f"Error: {str(e)}"

    history.append((message, response))
    return history


def get_workflow_visualization():
    """Generate workflow visualization from last execution."""
    trace = memory.get_last_trace()

    if not trace:
        return "No workflow yet. Send a message!", "", ""

    workflow_graph = """## 🔄 Agora Workflow Execution

```
User Query → InterpretQuery (AI + Memory)
                    ↓
"""

    intent = trace.get("intent", {})
    action = intent.get("action", "UNKNOWN")

    if action == "SEARCH_SYMBOL":
        workflow_graph += """         [SEARCH_SYMBOL]
                    ↓
         SearchSymbol (Alpha Vantage)
                    ↓
         GetQuoteParallel (Parallel API calls)
                    ↓
         GetOverview
                    ↓
         GenerateResponse (AI + Memory)
"""
    elif action == "GET_QUOTE":
        workflow_graph += """         [GET_QUOTE]
                    ↓
         GetQuoteParallel (Parallel API calls)
                    ↓
         GetOverview
                    ↓
         GenerateResponse (AI + Memory)
"""
    elif action == "GENERAL_CHAT":
        workflow_graph += """         [GENERAL_CHAT]
                    ↓
         GeneralChat (No API)
                    ↓
         GenerateResponse (AI + Memory)
"""
    else:
        workflow_graph += f"""         [{action}]
                    ↓
         GetOverview
                    ↓
         GenerateResponse (AI + Memory)
"""

    workflow_graph += "```"

    data_flow = f"""## 📊 Data Flow & Memory

**Query:** {trace.get('user_query', 'N/A')}

**Memory Context:** {trace.get('memory_context', 'None')}

**AI Interpretation:**
- Action: `{action}`
- Symbols: {intent.get('symbols', [])}
- Search: {intent.get('search_query', 'N/A')}

**Data Collected:**
"""

    search_results = trace.get("search_results", [])
    if search_results:
        data_flow += f"\n🔍 **Search:** {len(search_results)} companies\n"
        for sr in search_results[:3]:
            data_flow += f"  - {sr.get('symbol')}: {sr.get('name')}\n"

    quotes = trace.get("quotes", [])
    if quotes:
        data_flow += f"\n📈 **Quotes (Parallel):** {len(quotes)} fetched\n"
        for q in quotes:
            data_flow += f"  - {q.get('symbol')}: ${q.get('price')} ({q.get('change_percent')})\n"

    overviews = trace.get("overviews", [])
    if overviews:
        data_flow += f"\n📋 **Overviews:** {len(overviews)} fetched\n"
        for o in overviews:
            data_flow += f"  - {o.get('symbol')} | {o.get('sector')} | P/E: {o.get('pe_ratio')}\n"

    if not search_results and not quotes and not overviews:
        data_flow += "\n*No API calls (used memory/general knowledge)*\n"

    timing = f"""## ⏱️ Execution Performance

**Total:** {trace.get('total_time', 0):.2f}s

**Breakdown:**
- Intent Analysis: ~0.5s (OpenAI + Memory lookup)
"""

    if search_results:
        timing += "- Symbol Search: ~0.5s\n"
    if quotes:
        timing += f"- Parallel Quotes: ~0.5s ({len(quotes)} calls in parallel!)\n"
    if overviews:
        timing += f"- Overviews: ~{len(overviews) * 0.3:.1f}s\n"

    timing += "- Response Gen: ~1.0s (OpenAI + Memory)\n"

    return workflow_graph, data_flow, timing


def get_memory_state():
    """Get current memory state."""
    return f"""## 💾 Conversation Memory State

**Total Messages:** {len(memory.history)}

**Symbols Discussed:** {', '.join(list(memory.mentioned_symbols)) if memory.mentioned_symbols else 'None yet'}

**Last Intent:** {memory.last_intent or 'None'}

**Workflow Traces Stored:** {len(memory.workflow_traces)}

**Recent Context:**
{memory.get_context_string()}

---

### 📝 Recent Messages:
"""  + "\n".join([f"- {m['role']}: {m['content'][:100]}..." for m in memory.history[-5:]])


def get_node_log():
    """Get node execution log from telemetry."""
    try:
        with open("agora_workflow_traces.jsonl", "r") as f:
            lines = f.readlines()[-25:]

        if not lines:
            return "No telemetry yet"

        log = "## 📝 Agora Node Execution Log\n\n"

        for line in reversed(lines[:15]):
            try:
                trace = json.loads(line)
                name = trace.get("name", "")
                if any(n in name for n in ["Interpret", "Search", "Quote", "Overview", "Generate", "Chat"]):
                    attrs = trace.get("attributes", {})
                    duration = attrs.get("duration_ms", "N/A")
                    phase = attrs.get("agora.phase", "")
                    retry = attrs.get("retry_count", 0)

                    if phase:
                        retry_str = f" 🔄 retry {retry}" if retry > 0 else ""
                        log += f"- **{name}**: {duration}ms{retry_str}\n"
            except:
                continue

        return log

    except FileNotFoundError:
        return "No telemetry file yet. Make a query!"
    except Exception as e:
        return f"Error: {str(e)}"


def create_interface():
    """Create Gradio interface with dashboard."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Agora Stock Research Demo") as demo:
        gr.Markdown("""
        # 💬 Agora Framework Demo: Stock Research Chat
        ### Featuring: @agora_node decorator | TracedAsyncFlow | Conversation Memory | Parallel Processing

        **Try asking:**
        - "Tell me about Apple and Microsoft"
        - "Which biotech companies work on cancer?"
        - "Compare TSLA and F"
        - "Show me chip manufacturers"

        *Check the Dashboard tab to see Agora's workflow in action!*
        """)

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                label="Stock Research Assistant (with Memory)",
                height=500,
                show_copy_button=True,
                type="tuples"
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about stocks...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)

            gr.Markdown("""
            **💾 Memory-Enabled:** I remember what we discussed!
            **⚡ Parallel Processing:** Multiple API calls run simultaneously
            **📊 Telemetry:** Every node execution is automatically traced
            """)

            async def submit_msg(message, history):
                return await chat_response(message, history)

            msg.submit(submit_msg, [msg, chatbot], [chatbot])
            submit.click(submit_msg, [msg, chatbot], [chatbot])

        with gr.Tab("📊 Agora Workflow Dashboard"):
            gr.Markdown("""
            # 🎯 Agora Framework Live Dashboard

            Watch how Agora orchestrates workflows with:
            - **@agora_node decorator** for simple node creation
            - **TracedAsyncFlow** for workflow orchestration
            - **Automatic telemetry** with OpenTelemetry
            - **Persistent memory** across conversations
            - **Parallel execution** for efficiency
            """)

            with gr.Row():
                with gr.Column():
                    workflow_viz = gr.Markdown(value="Send a message first!")
                with gr.Column():
                    data_flow = gr.Markdown()

            with gr.Row():
                timing = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    node_log = gr.Markdown()
                with gr.Column():
                    memory_state = gr.Markdown()

            refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", size="lg")

            def refresh_all():
                wf, df, tm = get_workflow_visualization()
                nl = get_node_log()
                ms = get_memory_state()
                return wf, df, tm, nl, ms

            refresh_btn.click(
                refresh_all,
                outputs=[workflow_viz, data_flow, timing, node_log, memory_state]
            )

            gr.Markdown("""
            ---
            ## 🔍 Agora Features Demonstrated

            **1. @agora_node Decorator**
            Simple function → Workflow node with automatic tracing
            ```python
            @agora_node(name="FetchQuote", max_retries=2, wait=1)
            async def fetch_quote(shared):
                # Your code here
                return "next_action"
            ```

            **2. TracedAsyncFlow**
            Orchestrates async workflow with built-in telemetry
            ```python
            flow = TracedAsyncFlow("MyWorkflow")
            flow.start(node1)
            node1 - "action" >> node2
            ```

            **3. Persistent Memory**
            Maintains context across workflow executions

            **4. Parallel Processing**
            Multiple API calls execute simultaneously (see GetQuoteParallel)

            **5. OpenTelemetry Integration**
            Automatic tracing with prep/exec/post phases

            **6. Dynamic Routing**
            AI decides workflow path based on user intent

            **Try different queries to see different workflow paths!**
            """)

    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🎯 Agora Framework Demo: Stock Research Chat")
    print("="*60)
    print("\n✅ Features:")
    print("   • @agora_node decorator")
    print("   • TracedAsyncFlow orchestration")
    print("   • Conversation memory")
    print("   • Parallel API calls")
    print("   • Automatic telemetry")
    print("\n🚀 Launching on port 7862...\n")

    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7862,  # NEW PORT
        show_error=True
    )


if __name__ == "__main__":
    main()
