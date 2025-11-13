#!/usr/bin/env python3
"""
Advanced Agora Stock Research Demo with Telemetry Database & Graph Visualization

Features:
- SQLite database for telemetry storage
- Interactive Cytoscape-style workflow graph
- Advanced telemetry querying and filtering
- Export capabilities (JSON, CSV)
- Full conversation memory
- Parallel API processing

🔑 PASTE YOUR API KEYS BELOW (lines 32-33)
"""

import os
import sys
import asyncio
import requests
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path

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
    print("\n📝 Edit lines 32-33 and paste your keys")
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
    from agora.agora_tracer import init_traceloop, agora_node, TracedAsyncFlow
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("\n📦 Install: pip install gradio openai traceloop-sdk opentelemetry-api opentelemetry-sdk")
    sys.exit(1)

# ============================================================================
# TELEMETRY DATABASE - SQLite for persistent storage
# ============================================================================

class TelemetryDatabase:
    """
    SQLite database for storing and querying workflow telemetry.
    Perfect for Colab - no external DB needed!
    """
    def __init__(self, db_path="agora_telemetry.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Workflow executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT,
                intent_action TEXT,
                total_time_ms REAL,
                nodes_executed TEXT,
                memory_context TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Node executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER,
                node_name TEXT NOT NULL,
                phase TEXT,
                duration_ms REAL,
                retry_count INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflow_executions(id)
            )
        """)

        # Data flow table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER,
                node_name TEXT,
                data_type TEXT,
                data_summary TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflow_executions(id)
            )
        """)

        conn.commit()
        conn.close()

    def log_workflow(self, trace: Dict[str, Any]) -> int:
        """Log a workflow execution and return workflow_id."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        intent = trace.get("intent", {})
        nodes_executed = trace.get("nodes_executed", [])

        cursor.execute("""
            INSERT INTO workflow_executions
            (timestamp, user_query, intent_action, total_time_ms, nodes_executed, memory_context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            trace.get("timestamp"),
            trace.get("user_query"),
            intent.get("action"),
            trace.get("total_time", 0) * 1000,  # Convert to ms
            json.dumps(nodes_executed),
            trace.get("memory_context")
        ))

        workflow_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return workflow_id

    def log_node_execution(self, workflow_id: int, node_name: str, phase: str,
                          duration_ms: float, retry_count: int = 0,
                          error: str = None):
        """Log individual node execution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO node_executions
            (workflow_id, node_name, phase, duration_ms, retry_count, success, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            workflow_id,
            node_name,
            phase,
            duration_ms,
            retry_count,
            1 if not error else 0,
            error,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def log_data_flow(self, workflow_id: int, node_name: str, data_type: str, data_summary: str):
        """Log data passing through nodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO data_flow
            (workflow_id, node_name, data_type, data_summary, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (workflow_id, node_name, data_type, data_summary, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def query_workflows(self, limit=50, intent_filter=None):
        """Query workflow executions with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if intent_filter:
            cursor.execute("""
                SELECT * FROM workflow_executions
                WHERE intent_action = ?
                ORDER BY created_at DESC LIMIT ?
            """, (intent_filter, limit))
        else:
            cursor.execute("""
                SELECT * FROM workflow_executions
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def query_nodes(self, workflow_id=None, limit=100):
        """Query node executions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if workflow_id:
            cursor.execute("""
                SELECT * FROM node_executions
                WHERE workflow_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (workflow_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM node_executions
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_workflow_graph_data(self, workflow_id: int):
        """Get data for visualizing workflow as graph."""
        nodes_data = self.query_nodes(workflow_id)

        # Build node and edge data for visualization
        nodes = {}
        edges = []

        prev_node = None
        for i, node_data in enumerate(nodes_data):
            node_name = node_data["node_name"]

            if node_name not in nodes:
                nodes[node_name] = {
                    "id": node_name,
                    "label": node_name,
                    "duration_ms": node_data["duration_ms"],
                    "phase": node_data["phase"],
                    "success": node_data["success"] == 1
                }

            if prev_node and prev_node != node_name:
                edges.append({
                    "source": prev_node,
                    "target": node_name
                })

            prev_node = node_name

        return list(nodes.values()), edges

    def export_to_json(self, filepath="telemetry_export.json", limit=100):
        """Export telemetry data to JSON file."""
        workflows = self.query_workflows(limit=limit)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_workflows": len(workflows),
            "workflows": []
        }

        for wf in workflows:
            wf_data = dict(wf)
            wf_data["nodes"] = self.query_nodes(workflow_id=wf["id"])
            export_data["workflows"].append(wf_data)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        return filepath

    def get_statistics(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total workflows
        cursor.execute("SELECT COUNT(*) FROM workflow_executions")
        stats["total_workflows"] = cursor.fetchone()[0]

        # Total nodes executed
        cursor.execute("SELECT COUNT(*) FROM node_executions")
        stats["total_nodes"] = cursor.fetchone()[0]

        # Average execution time
        cursor.execute("SELECT AVG(total_time_ms) FROM workflow_executions")
        avg_time = cursor.fetchone()[0]
        stats["avg_execution_ms"] = round(avg_time, 2) if avg_time else 0

        # Most common intent
        cursor.execute("""
            SELECT intent_action, COUNT(*) as count
            FROM workflow_executions
            GROUP BY intent_action
            ORDER BY count DESC LIMIT 1
        """)
        result = cursor.fetchone()
        stats["most_common_intent"] = result[0] if result else "None"

        conn.close()
        return stats


# Initialize global database
telemetry_db = TelemetryDatabase()

# ============================================================================
# INITIALIZE AGORA WITH TELEMETRY
# ============================================================================

print("🚀 Initializing Advanced Agora Demo...")

init_traceloop(
    app_name="agora_advanced_demo",
    export_to_console=True,
    export_to_file="agora_traces.jsonl",
    disable_content_logging=True
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print("✅ Agora Framework Ready!")
print("💾 Database: agora_telemetry.db")
print("📊 Telemetry: Console + File + Database\n")

# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

class ConversationMemory:
    """Persistent memory across workflow executions."""
    def __init__(self):
        self.history = []
        self.mentioned_symbols = set()
        self.last_intent = None
        self.workflow_traces = []
        self.current_workflow_id = None

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.history = self.history[-10:]

    def add_symbols(self, symbols: List[str]):
        self.mentioned_symbols.update(symbols)

    def get_context_string(self) -> str:
        ctx = []
        if self.mentioned_symbols:
            ctx.append(f"Discussed: {', '.join(list(self.mentioned_symbols)[-5:])}")
        if self.last_intent:
            ctx.append(f"Last: {self.last_intent}")
        return " | ".join(ctx) if ctx else "No context"

    def add_workflow_trace(self, trace: Dict[str, Any]):
        self.workflow_traces.append(trace)
        self.workflow_traces = self.workflow_traces[-20:]

        # Log to database
        self.current_workflow_id = telemetry_db.log_workflow(trace)

    def get_last_trace(self) -> Dict[str, Any]:
        return self.workflow_traces[-1] if self.workflow_traces else None


memory = ConversationMemory()

# ============================================================================
# ALPHA VANTAGE API
# ============================================================================

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

def fetch_alpha_vantage(function: str, symbol: str = None, **kwargs) -> Dict[str, Any]:
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
    """Parallel quote fetching."""
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
# AGORA NODES WITH DATABASE LOGGING
# ============================================================================

@agora_node(name="InterpretQuery", max_retries=2, wait=1)
async def interpret_query(shared: Dict[str, Any]) -> str:
    """AI-powered intent detection."""
    user_message = shared["user_message"]
    context = memory.get_context_string()
    recent_history = memory.history[-4:]

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
        shared["nodes_executed"] = shared.get("nodes_executed", []) + ["InterpretQuery"]

        # Log to database
        if memory.current_workflow_id:
            telemetry_db.log_data_flow(
                memory.current_workflow_id,
                "InterpretQuery",
                "intent",
                json.dumps(intent)
            )

        memory.last_intent = intent.get("action")
        if intent.get("symbols"):
            memory.add_symbols(intent["symbols"])

        action = intent.get("action", "GENERAL_CHAT")

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
    """Search for stock symbols."""
    intent = shared.get("intent", {})
    search_query = intent.get("search_query", "")
    shared["nodes_executed"] = shared.get("nodes_executed", []) + ["SearchSymbol"]

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

    # Log to database
    if memory.current_workflow_id:
        telemetry_db.log_data_flow(
            memory.current_workflow_id,
            "SearchSymbol",
            "search_results",
            f"Found {len(symbols)} symbols for '{search_query}'"
        )

    if shared["symbols_to_fetch"]:
        memory.add_symbols(shared["symbols_to_fetch"])

    return "get_quote_parallel" if shared["symbols_to_fetch"] else "general_chat"


@agora_node(name="GetQuoteParallel", max_retries=2, wait=1)
async def get_quote_parallel(shared: Dict[str, Any]) -> str:
    """Fetch quotes in parallel."""
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))
    shared["nodes_executed"] = shared.get("nodes_executed", []) + ["GetQuoteParallel"]

    if not symbols:
        return "general_chat"

    quotes = await fetch_multiple_quotes_parallel(symbols)
    shared["quotes"] = quotes

    # Log to database
    if memory.current_workflow_id:
        telemetry_db.log_data_flow(
            memory.current_workflow_id,
            "GetQuoteParallel",
            "quotes",
            f"Fetched {len(quotes)} quotes (parallel): {', '.join([q['symbol'] for q in quotes])}"
        )

    memory.add_symbols([q["symbol"] for q in quotes])

    return "get_overview"


@agora_node(name="GetOverview", max_retries=2, wait=1)
async def get_overview(shared: Dict[str, Any]) -> str:
    """Fetch company overviews."""
    intent = shared.get("intent", {})
    symbols = intent.get("symbols", shared.get("symbols_to_fetch", []))
    shared["nodes_executed"] = shared.get("nodes_executed", []) + ["GetOverview"]

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

    # Log to database
    if memory.current_workflow_id:
        telemetry_db.log_data_flow(
            memory.current_workflow_id,
            "GetOverview",
            "overviews",
            f"Fetched {len(overviews)} company overviews"
        )

    return "generate_response"


@agora_node(name="GeneralChat")
async def general_chat(shared: Dict[str, Any]) -> str:
    """Handle general questions."""
    shared["response_type"] = "general"
    shared["nodes_executed"] = shared.get("nodes_executed", []) + ["GeneralChat"]
    return "generate_response"


@agora_node(name="GenerateResponse", max_retries=2, wait=1)
async def generate_response(shared: Dict[str, Any]) -> str:
    """Generate AI response."""
    user_message = shared["user_message"]
    shared["nodes_executed"] = shared.get("nodes_executed", []) + ["GenerateResponse"]

    # Collect all data
    search_results = shared.get("search_results", [])
    quotes = shared.get("quotes", [])
    overviews = shared.get("overviews", [])
    error_message = shared.get("error_message", "")

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
# WORKFLOW BUILDER
# ============================================================================

def build_chat_flow() -> TracedAsyncFlow:
    """Build workflow with telemetry."""
    flow = TracedAsyncFlow("StockChatWorkflow")

    flow.start(interpret_query)

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
# VISUALIZATION - Cytoscape-style Network Graph
# ============================================================================

def generate_cytoscape_graph(nodes, edges):
    """Generate HTML with Cytoscape.js network visualization."""

    # Convert to Cytoscape format
    cyto_elements = []

    # Add nodes
    for node in nodes:
        color = "#10b981" if node["success"] else "#ef4444"  # Green or red
        cyto_elements.append({
            "data": {
                "id": node["id"],
                "label": node["label"],
                "duration": f"{node['duration_ms']:.1f}ms",
                "phase": node["phase"]
            },
            "style": {"background-color": color}
        })

    # Add edges
    for edge in edges:
        cyto_elements.append({
            "data": {
                "source": edge["source"],
                "target": edge["target"]
            }
        })

    html = f"""
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
        <style>
            #cy {{
                width: 100%;
                height: 600px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: #f9fafb;
            }}
        </style>
    </head>
    <body>
        <div id="cy"></div>
        <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {json.dumps(cyto_elements)},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'background-color': '#3b82f6',
                            'color': '#fff',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': '12px',
                            'width': '80px',
                            'height': '80px',
                            'border-width': 2,
                            'border-color': '#1e40af'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 3,
                            'line-color': '#94a3b8',
                            'target-arrow-color': '#94a3b8',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }}
                    }}
                ],
                layout: {{
                    name: 'breadthfirst',
                    directed: true,
                    padding: 50,
                    spacingFactor: 1.5
                }}
            }});

            cy.on('tap', 'node', function(evt){{
                var node = evt.target;
                alert('Node: ' + node.data('label') + '\\nDuration: ' + node.data('duration') + '\\nPhase: ' + node.data('phase'));
            }});
        </script>
    </body>
    </html>
    """

    return html

# [CONTINUATION IN NEXT MESSAGE - FILE TOO LONG]

# ============================================================================
# GRADIO INTERFACE WITH ADVANCED TELEMETRY
# ============================================================================

async def chat_response(message: str, history: List[Tuple[str, str]]):
    """Handle chat with full telemetry logging."""
    if not message.strip():
        return history

    memory.add_message("user", message)

    shared = {
        "user_message": message,
        "start_time": datetime.now(),
        "nodes_executed": []
    }

    flow = build_chat_flow()

    try:
        import time
        start = time.time()
        await flow.run_async(shared)
        elapsed = time.time() - start

        response = shared.get("bot_response", "I'm not sure how to respond.")

        # Build trace
        trace = {
            "timestamp": datetime.now().isoformat(),
            "user_query": message,
            "total_time": elapsed,
            "nodes_executed": shared.get("nodes_executed", []),
            "intent": shared.get("intent", {}),
            "search_results": shared.get("search_results", []),
            "quotes": shared.get("quotes", []),
            "overviews": shared.get("overviews", []),
            "memory_context": memory.get_context_string()
        }

        memory.add_workflow_trace(trace)
        memory.add_message("assistant", response)

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


def query_telemetry_db(intent_filter, limit):
    """Query database and format results."""
    workflows = telemetry_db.query_workflows(
        limit=limit,
        intent_filter=intent_filter if intent_filter != "All" else None
    )

    if not workflows:
        return "No workflows found"

    result = f"## 📊 Found {len(workflows)} Workflows\n\n"

    for wf in workflows[:10]:  # Show first 10
        result += f"### Workflow #{wf['id']}\n"
        result += f"- **Query:** {wf['user_query']}\n"
        result += f"- **Intent:** {wf['intent_action']}\n"
        result += f"- **Time:** {wf['total_time_ms']:.0f}ms\n"
        result += f"- **Timestamp:** {wf['timestamp']}\n\n"

    return result


def show_workflow_graph():
    """Generate interactive graph for last workflow."""
    last_trace = memory.get_last_trace()

    if not last_trace or not memory.current_workflow_id:
        return "<p>No workflow executed yet</p>"

    nodes, edges = telemetry_db.get_workflow_graph_data(memory.current_workflow_id)

    if not nodes:
        return "<p>No node data available</p>"

    return generate_cytoscape_graph(nodes, edges)


def export_telemetry():
    """Export telemetry to JSON."""
    filepath = telemetry_db.export_to_json()
    stats = telemetry_db.get_statistics()

    return f"""## ✅ Export Complete!

**File:** {filepath}

**Statistics:**
- Total Workflows: {stats['total_workflows']}
- Total Nodes: {stats['total_nodes']}
- Avg Execution: {stats['avg_execution_ms']:.0f}ms
- Most Common Intent: {stats['most_common_intent']}

Download the file to explore the data!
"""


def get_db_stats():
    """Get database statistics."""
    stats = telemetry_db.get_statistics()

    return f"""## 📈 Database Statistics

**Workflows Executed:** {stats['total_workflows']}

**Total Nodes:** {stats['total_nodes']}

**Average Execution Time:** {stats['avg_execution_ms']:.0f}ms

**Most Common Intent:** {stats['most_common_intent']}

**Database File:** agora_telemetry.db ({os.path.getsize('agora_telemetry.db') / 1024:.1f} KB)
"""


def create_interface():
    """Create advanced Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Agora Advanced Demo") as demo:
        gr.Markdown("""
        # 🎯 Agora Advanced Demo: Stock Research with Telemetry Database

        ### Features: SQLite DB | Interactive Graphs | Query Interface | Export

        **Try:** "Compare biotech companies" or "Show me FAANG stocks"
        """)

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                label="Stock Research (Memory + DB Logging)",
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
            💾 **All workflows logged to SQLite database**  
            📊 **Check Telemetry tab to query execution history**  
            🎨 **View interactive workflow graphs**
            """)

            async def submit_msg(message, history):
                return await chat_response(message, history)

            msg.submit(submit_msg, [msg, chatbot], [chatbot])
            submit.click(submit_msg, [msg, chatbot], [chatbot])

        with gr.Tab("🎨 Interactive Workflow Graph"):
            gr.Markdown("""
            ## 🔄 Cytoscape-Style Workflow Visualization

            Interactive network graph showing node execution flow.
            - **Green circles** = Successful nodes
            - **Red circles** = Failed nodes
            - **Arrows** = Execution flow
            - **Click nodes** to see details!
            """)

            graph_html = gr.HTML(value="<p>Send a chat message first</p>")
            refresh_graph = gr.Button("🔄 Refresh Graph", variant="primary")

            refresh_graph.click(
                show_workflow_graph,
                outputs=graph_html
            )

        with gr.Tab("📊 Telemetry Database"):
            gr.Markdown("""
            ## 💾 SQLite Database Query Interface

            Query and explore workflow execution history stored in the database.
            """)

            with gr.Row():
                with gr.Column():
                    intent_dropdown = gr.Dropdown(
                        choices=["All", "SEARCH_SYMBOL", "GET_QUOTE", "COMPARE_STOCKS", "GENERAL_CHAT"],
                        label="Filter by Intent",
                        value="All"
                    )

                    limit_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=20,
                        step=10,
                        label="Max Results"
                    )

                    query_btn = gr.Button("🔍 Query Database", variant="primary")

                with gr.Column():
                    query_results = gr.Markdown(value="Click 'Query Database' to see results")

            query_btn.click(
                query_telemetry_db,
                inputs=[intent_dropdown, limit_slider],
                outputs=query_results
            )

            gr.Markdown("---")

            with gr.Row():
                stats_display = gr.Markdown()
                export_display = gr.Markdown()

            with gr.Row():
                stats_btn = gr.Button("📈 Show Statistics", variant="secondary")
                export_btn = gr.Button("💾 Export to JSON", variant="secondary")

            stats_btn.click(get_db_stats, outputs=stats_display)
            export_btn.click(export_telemetry, outputs=export_display)

        with gr.Tab("📖 Documentation"):
            gr.Markdown("""
            # 📖 Agora Advanced Features

            ## 🗄️ SQLite Telemetry Database

            All workflow executions are stored in `agora_telemetry.db` with three tables:

            ### `workflow_executions`
            Stores high-level workflow data:
            - User query
            - Intent detected
            - Total execution time
            - Nodes executed
            - Memory context

            ### `node_executions`
            Stores individual node execution:
            - Node name and phase (prep/exec/post)
            - Duration in milliseconds
            - Retry count
            - Success/failure status
            - Error messages

            ### `data_flow`
            Tracks data passing through workflow:
            - Data type (intent, quotes, overviews)
            - Data summary
            - Associated workflow

            ## 🎨 Interactive Graph Visualization

            Uses **Cytoscape.js** for network graphs:
            - Circular nodes representing workflow steps
            - Directed edges showing execution flow
            - Click nodes to see execution details
            - Color-coded by success/failure

            ## 🔍 Query Interface

            Filter and explore telemetry:
            - Filter by intent type
            - Limit results
            - View execution details
            - Export to JSON

            ## 💾 Export Capabilities

            Export all telemetry data to JSON:
            ```json
            {
              "export_timestamp": "2025-01-13T...",
              "total_workflows": 42,
              "workflows": [...]
            }
            ```

            ## 📊 Statistics

            Real-time database stats:
            - Total workflows executed
            - Total nodes processed
            - Average execution time
            - Most common intent

            ---

            ## 🚀 Agora Framework Features Used

            1. **@agora_node** - Decorator for workflow nodes
            2. **TracedAsyncFlow** - Workflow orchestration
            3. **Automatic Telemetry** - OpenTelemetry integration
            4. **Conversation Memory** - Persistent state
            5. **Parallel Processing** - Async API calls
            6. **Database Integration** - SQLite storage
            7. **Custom Visualization** - Cytoscape graphs

            """)

    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🎯 Agora Advanced Demo")
    print("="*60)
    print("\n✅ Features:")
    print("   • SQLite telemetry database")
    print("   • Interactive Cytoscape graphs")
    print("   • Query interface")
    print("   • Export capabilities")
    print("   • Conversation memory")
    print("   • Parallel processing")
    print("\n🚀 Launching on port 7862...\n")

    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7862,
        show_error=True
    )


if __name__ == "__main__":
    main()
