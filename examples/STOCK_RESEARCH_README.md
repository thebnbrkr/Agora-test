# 📊 Agora Stock Research Demo

A comprehensive demonstration of Agora's workflow capabilities with real-time stock market data from Alpha Vantage, AI analysis from OpenAI, and an interactive Gradio interface.

## 🎯 Features

### Core Agora Features
- **`@agora_node` Decorator**: Simple function-to-node conversion
- **TracedAsyncFlow**: Efficient async workflow orchestration
- **Built-in Telemetry**: Automatic OpenTelemetry/Traceloop integration
- **Error Handling**: Automatic retries and graceful fallbacks
- **Routing Logic**: Dynamic workflow paths based on node outputs

### Stock Research Features
- Real-time stock quotes and price data
- Company fundamentals and overview
- Historical price trends
- AI-powered stock analysis
- Interactive Gradio web interface

## 📦 Installation

### Option 1: Google Colab (Recommended for Quick Start)

1. Open the notebook: `stock_research_colab.ipynb`
2. Upload to Google Colab
3. Add your API keys to Colab secrets:
   - Go to the key icon (🔑) in the left sidebar
   - Add `OPENAI_API_KEY`
   - Add `ALPHA_VANTAGE_KEY`
4. Run all cells

### Option 2: Local Installation

```bash
# Clone Agora repository
git clone https://github.com/JerzyKultura/Agora.git
cd Agora

# Install dependencies
pip install -e .
pip install openai gradio requests traceloop-sdk

# Set environment variables
export OPENAI_API_KEY="your-openai-key-here"
export ALPHA_VANTAGE_KEY="your-alpha-vantage-key-here"

# Run the app
python examples/stock_research_app.py
```

## 🔑 Getting API Keys

### OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create new secret key
5. Copy and save securely

### Alpha Vantage API Key (Free!)
1. Go to [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Enter your email address
3. Receive API key instantly (no credit card required)
4. Free tier includes:
   - 5 API calls per minute
   - 100 API calls per day
   - Real-time and historical stock data

## 🚀 Usage

### Running the Standalone App

```bash
python stock_research_app.py
```

The Gradio interface will open at: `http://localhost:7860`

### Running in Google Colab

1. Open `stock_research_colab.ipynb` in Colab
2. Run all cells
3. Use the Gradio interface directly in the notebook

### Testing Individual Components

```python
import asyncio
from stock_research_app import build_stock_research_flow

# Test the workflow
async def test():
    shared = {"symbol": "AAPL"}
    flow = build_stock_research_flow()
    await flow.run_async(shared)
    print(shared)

asyncio.run(test())
```

## 📊 Architecture

### Workflow Graph

```
┌─────────────────┐
│ ValidateSymbol  │ ──error──> [HandleError] ──> END
└────────┬────────┘
         │ fetch_quote
         ▼
┌─────────────────┐
│   FetchQuote    │ ──error──> [HandleError] ──> END
└────────┬────────┘
         │ fetch_overview
         ▼
┌─────────────────┐
│ FetchOverview   │
└────────┬────────┘
         │ fetch_historical
         ▼
┌─────────────────┐
│FetchHistorical  │
└────────┬────────┘
         │ analyze
         ▼
┌─────────────────┐
│  AnalyzeStock   │ ──complete──> END
└─────────────────┘
```

### Node Descriptions

| Node | Purpose | API Calls | Retries |
|------|---------|-----------|---------|
| **ValidateSymbol** | Validate user input | 0 | 1 |
| **FetchQuote** | Get real-time price | Alpha Vantage GLOBAL_QUOTE | 2 |
| **FetchOverview** | Get company fundamentals | Alpha Vantage OVERVIEW | 2 |
| **FetchHistorical** | Get price history | Alpha Vantage TIME_SERIES_DAILY | 2 |
| **AnalyzeStock** | AI-powered analysis | OpenAI GPT-4o-mini | 2 |
| **HandleError** | Error handling | 0 | 1 |

## 🔧 Code Examples

### Example 1: Basic Node with Decorator

```python
from agora.agora_tracer import agora_node

@agora_node(name="FetchQuote", max_retries=2, wait=1)
async def fetch_quote(shared):
    """
    Simple async function converted to Agora node.
    - Automatic telemetry tracking
    - Automatic retry logic
    - Error handling built-in
    """
    symbol = shared["symbol"]

    # Fetch data
    data = await fetch_stock_data(symbol)

    # Store results
    shared["quote"] = data

    # Return routing action
    return "fetch_overview"  # Next node
```

### Example 2: Building the Flow

```python
from agora.agora_tracer import TracedAsyncFlow

# Create flow with automatic telemetry
flow = TracedAsyncFlow("StockResearch")

# Set starting node
flow.start(validate_symbol)

# Define routing with intuitive syntax
validate_symbol - "fetch_quote" >> fetch_quote
validate_symbol - "error" >> handle_error

fetch_quote - "fetch_overview" >> fetch_overview
fetch_overview - "fetch_historical" >> fetch_historical
fetch_historical - "analyze" >> analyze_stock

# Run asynchronously
await flow.run_async(shared)
```

### Example 3: Initializing Telemetry

```python
from agora.agora_tracer import init_traceloop

# One-time initialization
init_traceloop(
    app_name="stock_research",
    export_to_console=True,      # Print spans to console
    export_to_file="traces.jsonl", # Optional: save to file
    disable_content_logging=True  # Don't log API payloads
)

# Now all nodes automatically tracked!
```

## 📈 Telemetry Output

When you run the app, you'll see telemetry output like this:

```
✅ Traceloop initialized: stock_research_app
📊 All node executions will be traced automatically

{
    "name": "ValidateSymbol.prep",
    "timestamp": "2025-01-13T10:30:45.123Z",
    "duration_ms": 0.5,
    "attributes": {
        "agora.node": "ValidateSymbol",
        "agora.phase": "prep"
    }
}

{
    "name": "FetchQuote.exec",
    "timestamp": "2025-01-13T10:30:45.500Z",
    "duration_ms": 245.3,
    "attributes": {
        "agora.node": "FetchQuote",
        "agora.phase": "exec",
        "retry_count": 0
    }
}
```

## 🎨 Customization

### Adding New Data Sources

```python
@agora_node(name="FetchNews")
async def fetch_news(shared):
    """Add news sentiment analysis"""
    symbol = shared["symbol"]

    # Alpha Vantage News Sentiment API
    data = await fetch_alpha_vantage(
        "NEWS_SENTIMENT",
        tickers=symbol
    )

    shared["news"] = data
    return "analyze"

# Add to workflow
fetch_historical - "fetch_news" >> fetch_news
fetch_news - "analyze" >> analyze_stock
```

### Adding Technical Indicators

```python
@agora_node(name="FetchTechnicals")
async def fetch_technicals(shared):
    """Fetch RSI, MACD, SMA indicators"""
    symbol = shared["symbol"]

    # Fetch RSI
    rsi = await fetch_alpha_vantage("RSI", symbol=symbol, interval="daily")

    # Fetch MACD
    macd = await fetch_alpha_vantage("MACD", symbol=symbol, interval="daily")

    shared["technicals"] = {
        "rsi": rsi,
        "macd": macd
    }

    return "analyze"
```

### Custom Error Handling

```python
@agora_node(name="CustomErrorHandler")
async def custom_error_handler(shared):
    """Handle specific error types"""
    error = shared.get("error", "")

    if "rate limit" in error.lower():
        shared["result"] = "⏰ Rate limit hit. Try again in 1 minute."
    elif "not found" in error.lower():
        shared["result"] = "❌ Stock symbol not found."
    else:
        shared["result"] = f"❌ Error: {error}"

    return "complete"
```

## 🧪 Testing

### Unit Testing Individual Nodes

```python
import pytest

@pytest.mark.asyncio
async def test_validate_symbol():
    shared = {"symbol": "AAPL"}
    result = await validate_symbol._wrapped_func(shared)

    assert result == "fetch_quote"
    assert shared["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_invalid_symbol():
    shared = {"symbol": ""}
    result = await validate_symbol._wrapped_func(shared)

    assert result == "error"
    assert "error" in shared
```

### Integration Testing

```python
async def test_full_workflow():
    shared = {"symbol": "MSFT"}
    flow = build_stock_research_flow()

    await flow.run_async(shared)

    assert "quote" in shared
    assert "overview" in shared
    assert "analysis" in shared
    assert shared.get("success") == True
```

## 📚 Alpha Vantage API Reference

This demo uses the following Alpha Vantage endpoints:

### GLOBAL_QUOTE
- **Purpose**: Real-time price and volume
- **Rate Limit**: 5 calls/minute
- **Response**: Current price, change, volume, high/low

### OVERVIEW
- **Purpose**: Company fundamentals
- **Rate Limit**: 5 calls/minute
- **Response**: Market cap, P/E ratio, dividend yield, description

### TIME_SERIES_DAILY
- **Purpose**: Historical daily prices
- **Rate Limit**: 5 calls/minute
- **Response**: OHLCV data for last 100+ days

See [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/) for full details.

## 🔍 Troubleshooting

### "API rate limit reached"
- **Cause**: Free tier limit is 5 calls/minute
- **Solution**: Wait 60 seconds between queries or upgrade to premium

### "Invalid symbol"
- **Cause**: Stock symbol not found or invalid format
- **Solution**: Check symbol on financial sites (Yahoo Finance, Google Finance)

### "Connection timeout"
- **Cause**: Network issues or Alpha Vantage downtime
- **Solution**: Check internet connection, retry, or check status.alphavantage.co

### Telemetry not showing
- **Cause**: `init_traceloop()` not called before node execution
- **Solution**: Ensure telemetry initialization happens before creating nodes

## 🎯 Next Steps

### Enhancements to Try

1. **Add Charting**
   ```python
   import matplotlib.pyplot as plt

   @agora_node(name="CreateChart")
   async def create_chart(shared):
       historical = shared["historical"]
       # Plot price chart
       plt.plot(dates, prices)
       plt.savefig("chart.png")
       shared["chart"] = "chart.png"
       return "complete"
   ```

2. **Compare Multiple Stocks**
   ```python
   @agora_node(name="CompareStocks")
   async def compare_stocks(shared):
       symbols = shared["symbols"]  # ["AAPL", "MSFT", "GOOGL"]

       # Fetch data for all symbols
       results = await asyncio.gather(*[
           fetch_quote({"symbol": s}) for s in symbols
       ])

       shared["comparison"] = results
       return "analyze"
   ```

3. **Export Reports**
   ```python
   @agora_node(name="ExportReport")
   async def export_report(shared):
       # Generate PDF report
       report = generate_pdf(shared)
       shared["report_path"] = report
       return "complete"
   ```

## 📖 Resources

- **Agora Documentation**: [github.com/JerzyKultura/Agora](https://github.com/JerzyKultura/Agora)
- **Alpha Vantage API**: [alphavantage.co/documentation](https://www.alphavantage.co/documentation/)
- **OpenTelemetry Python**: [opentelemetry.io/docs/languages/python](https://opentelemetry.io/docs/languages/python/)
- **Traceloop SDK**: [github.com/traceloop/openllmetry](https://github.com/traceloop/openllmetry)
- **Gradio**: [gradio.app](https://www.gradio.app/)

## 🤝 Contributing

Found a bug or have an improvement? Please open an issue or submit a PR!

## 📄 License

This example is part of the Agora project and follows the same license.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Stock data and analysis should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.

---

**Built with ❤️ using Agora, Alpha Vantage, OpenAI, and Gradio**
