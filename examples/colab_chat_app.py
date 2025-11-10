# ==============================================================
# SIMPLE CHAT APP - Google Colab Ready
# ==============================================================
# Copy-paste this entire file into a Colab cell!
#
# This example shows TWO ways to build nodes:
# 1. @agora_node decorator (NEW! simpler!)
# 2. TracedAsyncNode classes (classic approach)
# ==============================================================

# Install dependencies
# !pip install -q openai traceloop-sdk nest_asyncio
# !pip install --force-reinstall -q git+https://github.com/thebnbrkr/Agora-test.git

import nest_asyncio
import os
import asyncio
import time
from openai import OpenAI

nest_asyncio.apply()

from agora.agora_tracer import (
    TracedAsyncNode,
    TracedAsyncFlow,
    init_traceloop,
    agora_node,  # ⭐ NEW DECORATOR!
)

# ==============================================================
# SETUP
# ==============================================================

# Initialize Traceloop
LOG_DIR = "/content/chat_telemetry"
os.makedirs(LOG_DIR, exist_ok=True)

init_traceloop(
    app_name="simple_chat",
    export_to_console=True,
    export_to_file=f"{LOG_DIR}/traces_{int(time.time())}.jsonl",
)

# Setup OpenAI
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # ⚠️ Replace with your key!
client = OpenAI()


# ==============================================================
# METHOD 1: Using @agora_node Decorator (SIMPLER!)
# ==============================================================
# The decorator wraps your function into a TracedAsyncNode.
# Your function receives the shared dict and returns the next action.
# ==============================================================


@agora_node(name="GetUserInput")
async def get_user_input(shared):
    """Get input from user - decorated function approach"""

    # Initialize chat on first turn
    if "messages" not in shared:
        shared["messages"] = []
        shared["turn"] = 0
        print("=" * 60)
        print("💬 CHAT READY - Type 'exit' to quit")
        print("=" * 60)
        print()

    # Get user input
    user_input = await asyncio.to_thread(input, "👤 You: ")
    user_input = user_input.strip()

    # Handle exit
    if user_input.lower() in ["exit", "quit", "bye"]:
        return "exit"

    # Store message
    shared["turn"] += 1
    shared["messages"].append({"role": "user", "content": user_input})

    # Return next action for routing
    return "respond"


@agora_node(name="GenerateResponse")
async def generate_response(shared):
    """Generate AI response - decorated function approach"""

    messages = shared["messages"]
    start = time.time()

    # Call OpenAI (Traceloop auto-instruments this!)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )

    latency = (time.time() - start) * 1000
    content = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens

    # Print response
    print(f"\n🤖 Bot: {content}")
    print(f"⏱️  {latency:.0f}ms | {tokens} tokens\n")

    # Store assistant message
    shared["messages"].append({"role": "assistant", "content": content})

    # Return next action
    return "input"


# ==============================================================
# METHOD 2: Using TracedAsyncNode Classes (Classic)
# ==============================================================
# This approach gives you more control with prep/exec/post phases.
# ==============================================================


class ExitNode(TracedAsyncNode):
    """Exit the chat gracefully"""

    async def exec_async(self, prep_res):
        print()
        print("=" * 60)
        print("👋 Thanks for chatting! Goodbye!")
        print("=" * 60)
        return None


# ==============================================================
# MAIN CHAT FLOW
# ==============================================================


async def run_chat():
    """Main chat flow with routing"""

    # Create nodes using BOTH approaches:
    # - Decorated functions (get_user_input, generate_response)
    # - Class instance (exit_node)
    input_node = get_user_input  # ⭐ This is already a TracedAsyncNode!
    response_node = generate_response  # ⭐ This too!
    exit_node = ExitNode("ChatExit")

    # Build flow with transitions
    flow = TracedAsyncFlow("ChatFlow")
    flow.start(input_node)

    # Define routing
    input_node - "respond" >> response_node
    input_node - "exit" >> exit_node
    response_node - "input" >> input_node

    # Run the flow
    shared = {}
    await flow.run_async(shared)

    # Print summary
    print()
    print("=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Total turns: {shared.get('turn', 0)}")
    print(f"Total messages: {len(shared.get('messages', []))}")
    print()
    print("🔍 Flow structure:")
    print(flow.to_mermaid())
    print("=" * 60)
    print()
    print(f"📁 Traces saved to: {LOG_DIR}")
    print("=" * 60)


# ==============================================================
# RUN IT!
# ==============================================================

# Run the chat
await run_chat()

# ==============================================================
# KEY FEATURES:
# ==============================================================
# ✅ @agora_node decorator - wrap any function into a node!
# ✅ Automatic OpenTelemetry tracing via Traceloop
# ✅ State management via shared dict
# ✅ Routing based on return values
# ✅ Mix decorators and classes freely
# ✅ Full telemetry export (console + file)
# ==============================================================
