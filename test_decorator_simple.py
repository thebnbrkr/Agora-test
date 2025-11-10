"""
Simple test without full traceloop initialization
Tests the decorator logic only
"""

import asyncio
import sys
import os

# Mock traceloop if not available
try:
    from traceloop.sdk import Traceloop
except ImportError:
    print("⚠️  Traceloop not installed, creating mock...")
    class MockTraceloop:
        @staticmethod
        def init(*args, **kwargs):
            pass
    sys.modules['traceloop'] = type(sys)('traceloop')
    sys.modules['traceloop.sdk'] = type(sys)('traceloop.sdk')
    sys.modules['traceloop'].sdk = sys.modules['traceloop.sdk']
    sys.modules['traceloop.sdk'].Traceloop = MockTraceloop

# Now import agora
from agora.agora_tracer import agora_node, TracedAsyncFlow, TracedAsyncNode

async def run_tests():
    print("=" * 60)
    print("Testing @agora_node Decorator (Without Full Traceloop)")
    print("=" * 60)

    # Test 1: Check decorator creates TracedAsyncNode
    print("\nTest 1: Decorator returns TracedAsyncNode instance...")

    @agora_node(name="TestNode")
    async def test_func(shared):
        shared["test"] = "works"
        return "next"

    assert isinstance(test_func, TracedAsyncNode), "Decorator should return TracedAsyncNode"
    assert test_func.name == "TestNode", "Node name should be set"
    print("✅ Decorator creates TracedAsyncNode correctly")

    # Test 2: Test async function execution
    print("\nTest 2: Async function execution...")

    @agora_node(name="AsyncTest")
    async def async_test(shared):
        shared["async"] = "success"
        await asyncio.sleep(0.01)
        # No return needed, flow ends

    flow = TracedAsyncFlow("TestFlow")
    flow.start(async_test)

    shared = {}

    # Run without full tracing (will use mock)
    try:
        result = await flow.run_async(shared)
        assert shared["async"] == "success"
        print("✅ Async function executes correctly")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    # Test 3: Test sync function execution
    print("\nTest 3: Sync function execution...")

    @agora_node(name="SyncTest")
    def sync_test(shared):
        shared["sync"] = "success"
        # No return needed, flow ends

    flow2 = TracedAsyncFlow("TestFlow2")
    flow2.start(sync_test)

    shared2 = {}
    result = await flow2.run_async(shared2)
    assert shared2["sync"] == "success"
    print("✅ Sync function executes correctly (via asyncio.to_thread)")

    # Test 4: Test chaining
    print("\nTest 4: Node chaining...")

    @agora_node(name="Node1")
    async def node1(shared):
        shared["value"] = 1
        # Return None for default routing (>> operator)

    @agora_node(name="Node2")
    async def node2(shared):
        shared["value"] += 10
        # Return None for default routing

    @agora_node(name="Node3")
    async def node3(shared):
        shared["value"] *= 2
        # No successor, flow ends

    flow3 = TracedAsyncFlow("ChainFlow")
    flow3.start(node1)
    node1 >> node2 >> node3  # >> creates "default" action routing

    shared3 = {}
    await flow3.run_async(shared3)
    assert shared3["value"] == 22, f"Expected 22, got {shared3['value']}"  # (1 + 10) * 2
    print("✅ Node chaining works correctly")

    # Test 5: Conditional routing
    print("\nTest 5: Conditional routing...")

    # Test route A
    @agora_node(name="RouterA")
    async def router_a(shared):
        if shared.get("route") == "A":
            return "route_a"
        return "route_b"

    @agora_node(name="HandlerA")
    async def handler_a(shared):
        shared["result"] = "A"
        # No return = flow ends here

    @agora_node(name="HandlerB1")
    async def handler_b1(shared):
        shared["result"] = "B"
        # No return = flow ends here

    flow_a = TracedAsyncFlow("RouteA")
    flow_a.start(router_a)
    router_a - "route_a" >> handler_a
    router_a - "route_b" >> handler_b1

    shared_a = {"route": "A"}
    await flow_a.run_async(shared_a)
    assert shared_a["result"] == "A", f"Expected A, got {shared_a.get('result')}"

    # Test route B (separate flow with separate nodes)
    @agora_node(name="RouterB")
    async def router_b(shared):
        if shared.get("route") == "A":
            return "route_a"
        return "route_b"

    @agora_node(name="HandlerA2")
    async def handler_a2(shared):
        shared["result"] = "A"

    @agora_node(name="HandlerB2")
    async def handler_b2(shared):
        shared["result"] = "B"

    flow_b = TracedAsyncFlow("RouteB")
    flow_b.start(router_b)
    router_b - "route_a" >> handler_a2
    router_b - "route_b" >> handler_b2

    shared_b = {"route": "B"}
    await flow_b.run_async(shared_b)
    assert shared_b["result"] == "B", f"Expected B, got {shared_b.get('result')}"

    print("✅ Conditional routing works correctly")

    # Test 6: Retry logic
    print("\nTest 6: Retry logic...")

    @agora_node(name="RetryNode", max_retries=3, wait=0.01)
    async def retry_node(shared):
        count = shared.get("attempts", 0)
        shared["attempts"] = count + 1

        if count < 2:
            raise ValueError("Intentional error for testing")

        shared["final_attempt"] = count
        return "done"

    flow_retry = TracedAsyncFlow("RetryFlow")
    flow_retry.start(retry_node)

    shared_retry = {}
    await flow_retry.run_async(shared_retry)
    assert shared_retry["attempts"] == 3, f"Should have retried 3 times, got {shared_retry['attempts']}"
    print("✅ Retry logic works correctly")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe @agora_node decorator is working perfectly!")
    print("\nKey features verified:")
    print("  ✓ Decorator creates TracedAsyncNode instances")
    print("  ✓ Async functions execute correctly")
    print("  ✓ Sync functions execute via asyncio.to_thread")
    print("  ✓ Node chaining works")
    print("  ✓ Conditional routing works")
    print("  ✓ Retry logic works")
    print("\nYou can now use @agora_node to wrap ANY function!")


if __name__ == "__main__":
    asyncio.run(run_tests())
