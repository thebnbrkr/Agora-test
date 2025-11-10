# PR Title:
**Add production telemetry and @agora_node decorator for simplified agent development**

---

# PR Description:

## 🎯 Overview

This PR adds production-grade observability and a decorator-based API to Agora, making it dramatically easier to build and debug multi-agent workflows.

## ✨ What's New

### 1. **Production Telemetry (`agora/agora_tracer.py`)**
- New `TracedAsyncNode` and `TracedAsyncFlow` classes with built-in OpenTelemetry tracing
- Automatic instrumentation of node lifecycle (prep/exec/post phases)
- LLM call tracking via Traceloop SDK integration (tokens, latency, model info)
- Export to console, files, or any OpenTelemetry backend (Datadog, Honeycomb, etc.)

**Key features:**
```python
from agora.agora_tracer import init_traceloop, TracedAsyncNode

# Initialize once at startup
init_traceloop(
    app_name="my_app",
    export_to_console=True,
    export_to_file="traces.jsonl"
)

# All nodes automatically traced
class MyAgent(TracedAsyncNode):
    async def exec_async(self, prep_res):
        # Automatically captures timing, errors, routing
        return await process(prep_res)
```

### 2. **@agora_node Decorator (Zero Boilerplate)**
- Wrap existing async functions into TracedAsyncNode with a single decorator
- No class inheritance, no framework restructuring required
- Just add `@agora_node` to your functions

**Before (class-based approach):**
```python
class MyAgent(TracedAsyncNode):
    async def prep_async(self, shared):
        return shared["input"]

    async def exec_async(self, prep_res):
        result = await openai.call(prep_res)
        return result

    async def post_async(self, shared, prep_res, exec_res):
        shared["result"] = exec_res
        return "next"
```

**After (decorator approach):**
```python
@agora_node(name="MyAgent")
async def my_agent(shared):
    result = await openai.call(shared["input"])
    shared["result"] = result
    return "next"  # Controls routing
```

### 3. **Production-Ready Examples**
- `examples/colab_chat_minimal.py` - Minimal 60-line chat app showing decorator usage
- `examples/colab_chat_app.py` - Full example demonstrating both approaches (decorator + classes)
- Both examples are Google Colab ready (copy-paste and run)

## 🔧 Technical Details

### Dependencies
- `opentelemetry-api` / `opentelemetry-sdk` (telemetry standard)
- `traceloop-sdk` (automatic LLM instrumentation)

### API Design
The decorator implementation:
1. Wraps your function into a `TracedAsyncNode` instance
2. Passes `shared` dict as parameter (state management)
3. Return value controls routing (e.g., `return "next"` routes to next node)
4. Supports both async and sync functions
5. Optional parameters: `name`, `max_retries`, `wait`

### Telemetry Captured
- **Node-level:** Duration, status, retry count, routing decisions
- **LLM calls:** Model, tokens (prompt/completion/total), latency, temperature
- **Errors:** Full exception traces with context
- **Flow-level:** End-to-end workflow timing

## 📊 Example Telemetry Output

```json
{
  "name": "GenerateResponse.exec",
  "attributes": {
    "agora.node": "GenerateResponse",
    "agora.phase": "exec",
    "duration_ms": 1011.04,
    "retry_count": 0
  }
}
```

```json
{
  "name": "openai.chat",
  "attributes": {
    "gen_ai.request.model": "gpt-4o-mini",
    "llm.usage.total_tokens": 181,
    "gen_ai.usage.completion_tokens": 150,
    "gen_ai.usage.prompt_tokens": 31
  }
}
```

## 🚀 Why This Matters

### Developer Experience
- **Minimal code changes:** Just add `@agora_node` decorator
- **No framework lock-in:** Keep your existing async functions
- **Immediate observability:** No separate instrumentation needed

### Production Readiness
- **Standard telemetry:** OpenTelemetry (not proprietary format)
- **Works with existing tools:** Datadog, Honeycomb, Grafana, etc.
- **Debug in production:** Full traces of agent execution paths
- **Cost tracking:** Automatic token/cost monitoring

## 🎯 Use Cases

Perfect for:
- Multi-agent RAG pipelines
- Research/analysis agents with multiple steps
- Production agent systems requiring observability
- Teams moving from prototypes to production

## 🧪 Testing

Run the examples to verify:
```bash
# Install dependencies
pip install openai traceloop-sdk nest_asyncio

# Install Agora
pip install git+https://github.com/thebnbrkr/Agora-test.git

# Run minimal example
python examples/colab_chat_minimal.py
```

## 📝 Breaking Changes

**None.** This is purely additive:
- Existing `AsyncNode` and `AsyncFlow` classes unchanged
- New traced classes are opt-in (`TracedAsyncNode` vs `AsyncNode`)
- Decorator is optional (can still use class-based approach)

## 🔮 Future Enhancements

Potential follow-ups:
- TypeScript/JavaScript SDK
- Visual workflow builder integration
- Cost optimization routing (route to cheaper models when possible)
- Distributed tracing across services

---

## 📦 Files Changed

**New files:**
- `agora/agora_tracer.py` - Core telemetry module
- `examples/colab_chat_minimal.py` - Minimal decorator example
- `examples/colab_chat_app.py` - Full-featured example

**Modified files:**
- None (fully backward compatible)

---

## ✅ Checklist

- [x] Code follows existing style
- [x] Fully backward compatible
- [x] Working examples included
- [x] Documentation in code (docstrings)
- [x] Tested with OpenAI API
- [x] Ready for Google Colab demos

---

## 🙏 Motivation

This PR positions Agora as a production-grade framework for AI agent orchestration with zero-friction observability. Key differentiators vs competitors (LangGraph, CrewAI):

1. **Built-in observability** (not a paid add-on)
2. **Minimal code changes** (decorator vs framework restructuring)
3. **OpenTelemetry standard** (not proprietary format)
4. **Production-focused** (not research-oriented)

Perfect timing for companies moving agent prototypes to production.
