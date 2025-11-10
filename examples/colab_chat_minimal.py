# ==============================================================
# MINIMAL CHAT APP - Using @agora_node Decorator
# ==============================================================
# This is the SIMPLEST possible chat app with Agora + Traceloop
# Copy-paste into Colab and run!
# ==============================================================

# !pip install -q openai traceloop-sdk nest_asyncio
# !pip install --force-reinstall -q git+https://github.com/thebnbrkr/Agora-test.git

import nest_asyncio, os, asyncio, time
from openai import OpenAI

nest_asyncio.apply()

from agora.agora_tracer import TracedAsyncFlow, init_traceloop, agora_node

# ==============================================================
# 1. INITIALIZE
# ==============================================================

init_traceloop(app_name="minimal_chat", export_to_console=True)
os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # ⚠️ Your key here!
client = OpenAI()


# ==============================================================
# 2. DEFINE NODES - Just Functions with @agora_node!
# ==============================================================


@agora_node(name="Input")
async def chat_input(shared):
    """Get user input"""
    if "messages" not in shared:
        shared["messages"] = []
        print("💬 Chat ready! Type 'exit' to quit.\n")

    user_text = await asyncio.to_thread(input, "You: ")

    if user_text.strip().lower() == "exit":
        return "exit"

    shared["messages"].append({"role": "user", "content": user_text})
    return "respond"


@agora_node(name="AI")
async def chat_ai(shared):
    """Generate AI response"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=shared["messages"],
        max_tokens=100,
    )

    content = response.choices[0].message.content
    print(f"\nBot: {content}\n")

    shared["messages"].append({"role": "assistant", "content": content})
    return "input"


@agora_node(name="Exit")
async def chat_exit(shared):
    """Exit gracefully"""
    print("👋 Goodbye!")
    return None


# ==============================================================
# 3. BUILD FLOW
# ==============================================================


async def run():
    flow = TracedAsyncFlow("Chat")
    flow.start(chat_input)

    # Route based on return values
    chat_input - "respond" >> chat_ai
    chat_input - "exit" >> chat_exit
    chat_ai - "input" >> chat_input

    # Run
    await flow.run_async({})


# ==============================================================
# 4. RUN IT!
# ==============================================================

await run()

# ==============================================================
# That's it! 🎉
# - No classes needed!
# - Just decorate your functions with @agora_node
# - Return values control routing
# - Full OpenTelemetry tracing included
# ==============================================================
