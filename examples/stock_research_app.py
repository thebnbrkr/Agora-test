#!/usr/bin/env python3
"""
Stock Research Application using Agora + Alpha Vantage + OpenAI + Gradio

This demonstrates:
- Agora's @agora_node decorator for workflow creation
- Built-in telemetry with Traceloop/OpenTelemetry
- Alpha Vantage API for real-time stock data
- Gradio interface for user interaction
- Efficient async processing with TracedAsyncFlow

Usage:
    python stock_research_app.py

Environment variables required:
    OPENAI_API_KEY - Your OpenAI API key
    ALPHA_VANTAGE_KEY - Your Alpha Vantage API key (free from alphavantage.co)
"""

import os
import sys
import asyncio
import requests
import json
from datetime import datetime
from typing import Dict, Any

# Check for required dependencies
try:
    import gradio as gr
    from openai import AsyncOpenAI
    from agora.agora_tracer import init_traceloop, agora_node, TracedAsyncFlow
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install required packages:")
    print("pip install gradio openai traceloop-sdk opentelemetry-api opentelemetry-sdk")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

# Get API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY")

if not OPENAI_API_KEY or not ALPHA_VANTAGE_KEY:
    print("❌ Missing API keys!")
    print("\n🔑 Set the following environment variables:")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ALPHA_VANTAGE_KEY='your-alpha-vantage-key'")
    print("\n💡 Get a free Alpha Vantage key at: https://www.alphavantage.co/support/#api-key")
    sys.exit(1)


# ============================================================================
# INITIALIZE TELEMETRY
# ============================================================================

print("🚀 Initializing Agora with telemetry...")

init_traceloop(
    app_name="stock_research_app",
    export_to_console=True,  # Show telemetry in console
    disable_content_logging=True  # Keep API calls private
)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print("✅ Telemetry initialized!")
print("📊 All node executions will be traced automatically\n")


# ============================================================================
# ALPHA VANTAGE API HELPERS
# ============================================================================

def fetch_alpha_vantage(function: str, symbol: str = None, **kwargs) -> Dict[str, Any]:
    """
    Efficient Alpha Vantage API caller with error handling.

    Args:
        function: Alpha Vantage function name (e.g., GLOBAL_QUOTE)
        symbol: Stock symbol (optional, depending on function)
        **kwargs: Additional parameters

    Returns:
        Dict containing API response or error
    """
    params = {
        "function": function,
        "apikey": ALPHA_VANTAGE_KEY,
        **kwargs
    }

    if symbol:
        params["symbol"] = symbol

    try:
        response = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            return {"error": data["Error Message"]}
        if "Note" in data:  # Rate limit
            return {"error": "API rate limit reached. Please wait a minute."}

        return data
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# STOCK RESEARCH NODES (using @agora_node decorator)
# ============================================================================

@agora_node(name="ValidateSymbol")
async def validate_symbol(shared: Dict[str, Any]) -> str:
    """
    Validate the stock symbol from user input.

    Returns routing action: "fetch_quote" or "error"
    """
    symbol = shared.get("symbol", "").strip().upper()

    if not symbol:
        shared["error"] = "Please enter a stock symbol"
        return "error"

    if len(symbol) > 10 or not symbol.isalpha():
        shared["error"] = f"Invalid symbol: {symbol}"
        return "error"

    shared["symbol"] = symbol
    shared["timestamp"] = datetime.now().isoformat()
    return "fetch_quote"


@agora_node(name="FetchQuote", max_retries=2, wait=1)
async def fetch_quote(shared: Dict[str, Any]) -> str:
    """
    Fetch real-time stock quote from Alpha Vantage.

    Returns routing action: "fetch_overview" or "error"
    """
    symbol = shared["symbol"]

    # Use asyncio.to_thread for sync API call
    data = await asyncio.to_thread(
        fetch_alpha_vantage,
        "GLOBAL_QUOTE",
        symbol=symbol
    )

    if "error" in data:
        shared["error"] = data["error"]
        return "error"

    quote = data.get("Global Quote", {})
    if not quote:
        shared["error"] = f"No quote data found for {symbol}"
        return "error"

    shared["quote"] = {
        "price": quote.get("05. price", "N/A"),
        "change": quote.get("09. change", "N/A"),
        "change_percent": quote.get("10. change percent", "N/A"),
        "volume": quote.get("06. volume", "N/A"),
        "high": quote.get("03. high", "N/A"),
        "low": quote.get("04. low", "N/A")
    }

    return "fetch_overview"


@agora_node(name="FetchOverview", max_retries=2, wait=1)
async def fetch_overview(shared: Dict[str, Any]) -> str:
    """
    Fetch company fundamentals and overview.

    Returns routing action: "fetch_historical"
    """
    symbol = shared["symbol"]

    data = await asyncio.to_thread(
        fetch_alpha_vantage,
        "OVERVIEW",
        symbol=symbol
    )

    if "error" in data:
        # Non-critical error - continue without overview
        shared["overview"] = {"error": data["error"]}
    else:
        shared["overview"] = {
            "name": data.get("Name", "N/A"),
            "sector": data.get("Sector", "N/A"),
            "industry": data.get("Industry", "N/A"),
            "market_cap": data.get("MarketCapitalization", "N/A"),
            "pe_ratio": data.get("PERatio", "N/A"),
            "dividend_yield": data.get("DividendYield", "N/A"),
            "52week_high": data.get("52WeekHigh", "N/A"),
            "52week_low": data.get("52WeekLow", "N/A"),
            "description": data.get("Description", "N/A")[:500]  # Truncate
        }

    return "fetch_historical"


@agora_node(name="FetchHistorical", max_retries=2, wait=1)
async def fetch_historical(shared: Dict[str, Any]) -> str:
    """
    Fetch recent daily price history.

    Returns routing action: "analyze"
    """
    symbol = shared["symbol"]

    data = await asyncio.to_thread(
        fetch_alpha_vantage,
        "TIME_SERIES_DAILY",
        symbol=symbol,
        outputsize="compact"  # Last 100 days
    )

    if "error" in data:
        shared["historical"] = {"error": data["error"]}
    else:
        time_series = data.get("Time Series (Daily)", {})
        # Get last 10 days
        recent = dict(list(time_series.items())[:10])
        shared["historical"] = recent

    return "analyze"


@agora_node(name="AnalyzeStock", max_retries=2, wait=1)
async def analyze_stock(shared: Dict[str, Any]) -> str:
    """
    Generate AI-powered stock analysis using OpenAI.

    Returns routing action: "complete"
    """
    symbol = shared["symbol"]
    quote = shared.get("quote", {})
    overview = shared.get("overview", {})

    # Build analysis prompt
    prompt = f"""Analyze the following stock data for {symbol}:

**Current Quote:**
- Price: ${quote.get('price', 'N/A')}
- Change: {quote.get('change', 'N/A')} ({quote.get('change_percent', 'N/A')})
- Volume: {quote.get('volume', 'N/A')}
- Day Range: ${quote.get('low', 'N/A')} - ${quote.get('high', 'N/A')}

**Company Overview:**
- Name: {overview.get('name', 'N/A')}
- Sector: {overview.get('sector', 'N/A')}
- Industry: {overview.get('industry', 'N/A')}
- Market Cap: {overview.get('market_cap', 'N/A')}
- P/E Ratio: {overview.get('pe_ratio', 'N/A')}
- Dividend Yield: {overview.get('dividend_yield', 'N/A')}
- 52-Week Range: ${overview.get('52week_low', 'N/A')} - ${overview.get('52week_high', 'N/A')}

Provide a brief analysis (3-4 paragraphs) covering:
1. Current price action and momentum
2. Valuation metrics (P/E, market cap)
3. Key strengths and potential risks
4. Overall outlook (neutral, avoid specific buy/sell recommendations)

Keep it concise and professional."""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )

        analysis = response.choices[0].message.content
        shared["analysis"] = analysis
        shared["success"] = True

    except Exception as e:
        shared["analysis"] = f"Analysis failed: {str(e)}"
        shared["success"] = True  # Still show results

    return "complete"


@agora_node(name="HandleError")
async def handle_error(shared: Dict[str, Any]) -> str:
    """
    Handle any errors that occurred during the workflow.

    Returns routing action: "complete"
    """
    error = shared.get("error", "Unknown error")
    shared["result"] = f"❌ Error: {error}"
    return "complete"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_stock_research_flow() -> TracedAsyncFlow:
    """
    Build the complete stock research workflow with automatic telemetry.

    Returns:
        TracedAsyncFlow instance ready to run
    """
    # Create flow with telemetry
    flow = TracedAsyncFlow("StockResearch")

    # Set starting node
    flow.start(validate_symbol)

    # Build the flow graph
    validate_symbol - "fetch_quote" >> fetch_quote
    validate_symbol - "error" >> handle_error

    fetch_quote - "fetch_overview" >> fetch_overview
    fetch_quote - "error" >> handle_error

    fetch_overview - "fetch_historical" >> fetch_historical

    fetch_historical - "analyze" >> analyze_stock

    # All paths lead to completion
    # (flow ends when no edges are defined for the returned action)

    return flow


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

async def research_stock(symbol: str):
    """
    Main function called by Gradio to research a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Tuple of (quote_text, overview_text, analysis_text)
    """
    # Create shared state
    shared = {"symbol": symbol}

    # Build and run flow
    flow = build_stock_research_flow()

    try:
        await flow.run_async(shared)
    except Exception as e:
        return f"❌ Workflow error: {str(e)}", None, None

    # Check for errors
    if "error" in shared and not shared.get("success"):
        return f"❌ {shared['error']}", None, None

    # Format results
    quote = shared.get("quote", {})
    overview = shared.get("overview", {})
    analysis = shared.get("analysis", "No analysis available")

    # Build quote summary
    quote_text = f"""## 📈 {symbol} - Live Quote

**Price:** ${quote.get('price', 'N/A')}
**Change:** {quote.get('change', 'N/A')} ({quote.get('change_percent', 'N/A')})
**Volume:** {quote.get('volume', 'N/A')}
**Day Range:** ${quote.get('low', 'N/A')} - ${quote.get('high', 'N/A')}
"""

    # Build company overview
    overview_text = f"""## 🏢 Company Overview

**Name:** {overview.get('name', 'N/A')}
**Sector:** {overview.get('sector', 'N/A')}
**Industry:** {overview.get('industry', 'N/A')}
**Market Cap:** ${overview.get('market_cap', 'N/A')}
**P/E Ratio:** {overview.get('pe_ratio', 'N/A')}
**Dividend Yield:** {overview.get('dividend_yield', 'N/A')}
**52-Week Range:** ${overview.get('52week_low', 'N/A')} - ${overview.get('52week_high', 'N/A')}

---

{overview.get('description', '')[:300]}...
"""

    # Build AI analysis
    analysis_text = f"""## 🤖 AI Analysis

{analysis}

---

*Analysis generated by GPT-4o-mini using Alpha Vantage data*
"""

    return quote_text, overview_text, analysis_text


def create_interface():
    """
    Create the Gradio interface for stock research.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Stock Research with Agora") as demo:
        gr.Markdown("""
        # 📊 Stock Research Demo
        ### Powered by Agora + Alpha Vantage + OpenAI

        Enter a stock symbol (e.g., AAPL, MSFT, TSLA) to get:
        - Real-time quote and price data
        - Company fundamentals and overview
        - AI-powered analysis

        *All operations are traced with OpenTelemetry - check console output!*
        """)

        with gr.Row():
            symbol_input = gr.Textbox(
                label="Stock Symbol",
                placeholder="Enter symbol (e.g., AAPL)",
                value="AAPL"
            )
            submit_btn = gr.Button("🔍 Research Stock", variant="primary")

        with gr.Row():
            with gr.Column():
                quote_output = gr.Markdown(label="Quote")
            with gr.Column():
                overview_output = gr.Markdown(label="Overview")

        with gr.Row():
            analysis_output = gr.Markdown(label="Analysis")

        gr.Markdown("""
        ---
        **Note:** This demo uses Alpha Vantage's free tier (5 API calls/minute, 100/day).
        Check the console for telemetry traces showing node execution times and spans!
        """)

        # Wire up the button
        submit_btn.click(
            fn=research_stock,
            inputs=[symbol_input],
            outputs=[quote_output, overview_output, analysis_output]
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for the application.
    """
    print("\n" + "="*60)
    print("📊 Stock Research App - Powered by Agora")
    print("="*60)
    print("\n🔧 Configuration:")
    print(f"   • OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"   • Alpha Vantage: {'✅ Configured' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
    print(f"   • Telemetry: ✅ Enabled (OpenTelemetry + Traceloop)")
    print("\n🚀 Starting Gradio interface...\n")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True for public sharing
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,  # Default Gradio port
        show_error=True
    )


if __name__ == "__main__":
    main()
