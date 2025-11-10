"""
Demo: Using @agora_node decorator to build a multi-agent system

This shows how EASY it is to convert existing functions into Agora nodes.
Just add @agora_node and you get:
- Automatic tracing with Traceloop/OpenTelemetry
- Retry logic
- Error handling
- Flow orchestration
"""

import asyncio
from agora.agora_tracer import init_traceloop, agora_node, TracedAsyncFlow

# Initialize tracing (do this once at startup)
init_traceloop(app_name="decorator_demo", export_to_console=True)


# ==============================================================
# OPTION 1: Decorator on Async Functions
# ==============================================================

@agora_node(name="InputHandler")
async def handle_input(shared):
    """Get user input - wrapped in Agora node with just one line!"""
    user_input = await asyncio.to_thread(input, "You: ")
    shared["user_input"] = user_input

    # Return action for routing
    if user_input.lower() in ["quit", "exit"]:
        return "quit"
    return "next"


@agora_node(name="ProcessAgent", max_retries=3, wait=1)
async def process_message(shared):
    """Process the message - this could be OpenAI, Anthropic, etc."""
    user_input = shared["user_input"]

    # Simulate API call
    await asyncio.sleep(0.1)

    # Your existing code here - no changes needed!
    response = f"Processed: {user_input.upper()}"

    shared["response"] = response
    return "next"


@agora_node(name="OutputHandler")
async def display_output(shared):
    """Display the response"""
    response = shared.get("response", "No response")
    print(f"Agent: {response}\n")
    return "loop"  # Go back to start


# ==============================================================
# OPTION 2: Decorator on Sync Functions (runs in thread)
# ==============================================================

@agora_node(name="SyncProcessor")
def sync_function(shared):
    """This is a regular sync function - Agora handles it!"""
    # Your existing sync code works fine
    data = shared.get("data", "")
    result = data.upper()  # Any sync operation
    shared["result"] = result
    return "next"


# ==============================================================
# BUILD THE FLOW - Same as Before!
# ==============================================================

async def main():
    # Create flow
    flow = TracedAsyncFlow("ChatLoop")

    # Set start node
    flow.start(handle_input)

    # Connect with routing
    handle_input - "next" >> process_message
    handle_input - "quit" >> None  # End flow
    process_message >> display_output
    display_output - "loop" >> handle_input  # Loop back

    # Run the flow
    print("Multi-Agent Chat (type 'quit' to exit)")
    print("=" * 50)

    shared = {}
    await flow.run_async(shared)

    print("\n✅ Flow completed!")


# ==============================================================
# COMPARISON: Before vs After
# ==============================================================

"""
BEFORE (Old Way):
-----------------
class HandleInput(TracedAsyncNode):
    def __init__(self):
        super().__init__("InputHandler")

    async def exec_async(self, prep_res):
        user_input = await asyncio.to_thread(input, "You: ")
        return user_input

    async def post_async(self, shared, prep_res, exec_res):
        shared["user_input"] = exec_res
        if exec_res.lower() in ["quit", "exit"]:
            return "quit"
        return "next"

handle_input = HandleInput()


AFTER (With Decorator):
-----------------------
@agora_node(name="InputHandler")
async def handle_input(shared):
    user_input = await asyncio.to_thread(input, "You: ")
    shared["user_input"] = user_input

    if user_input.lower() in ["quit", "exit"]:
        return "quit"
    return "next"


That's it! 10 lines vs 15 lines, way more readable, and your existing code works!
"""


if __name__ == "__main__":
    asyncio.run(main())
