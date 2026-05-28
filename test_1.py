# Tests for BufferMemory and Chatbot integration.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memories.buffer_memory import BufferMemory

_passed = 0
_failed = 0


def ok(msg: str) -> None:
    global _passed
    _passed += 1
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    global _failed
    _failed += 1
    print(f"  ✗  {msg}")


def test_overflow():
    print("\n-- overflow --")
    mem = BufferMemory(max_messages=4)

    for i in range(6):
        mem.add("user" if i % 2 == 0 else "assistant", f"msg {i}")

    s = mem.get_stats()
    if len(mem) == 4:
        ok(f"window capped at 4 (added 6)")
    else:
        fail(f"expected size=4, got {len(mem)}")

    if s["overflow_count"] == 2:
        ok("overflow_count=2")
    else:
        fail(f"expected overflow_count=2, got {s['overflow_count']}")

    ctx = mem.get_context()
    if ctx[0]["content"] == "msg 2":
        ok("oldest surviving message is correct")
    else:
        fail(f"unexpected oldest: {ctx[0]['content']}")


def test_clear():
    print("\nclear")
    mem = BufferMemory(max_messages=6)
    mem.add("user", "hello")
    mem.add("assistant", "hi")
    mem.clear()

    s = mem.get_stats()
    checks = [
        (len(mem) == 0,         "size=0 after clear"),
        (s["total_added"] == 0, "total_added reset"),
        (s["overflow_count"] == 0, "overflow_count reset"),
    ]
    for cond, msg in checks:
        ok(msg) if cond else fail(msg)


def test_validation():
    print("\ninput validation")
    mem = BufferMemory(max_messages=6)

    for bad_role in ["admin", "system", "", "USER"]:
        try:
            mem.add(bad_role, "text")
            fail(f"should reject role='{bad_role}'")
        except ValueError:
            ok(f"rejected role='{bad_role}'")

    for bad_content in ["", "   ", "\t\n"]:
        try:
            mem.add("user", bad_content)
            fail(f"should reject empty content {repr(bad_content)}")
        except ValueError:
            ok(f"rejected empty content {repr(bad_content)}")

    try:
        BufferMemory(max_messages=1)
        fail("should reject max_messages=1")
    except ValueError:
        ok("rejected max_messages=1")


def test_context_size():
    print("\ncontext size estimation")
    mem = BufferMemory(max_messages=10)
    mem.add("user", "a" * 400)  
    size = mem.get_context_size()
    if 80 <= size <= 120:
        ok(f"400-char message → {size} tokens (expected ~100)")
    else:
        fail(f"400-char message → {size} tokens, expected 80-120")

    mem.clear()
    if mem.get_context_size() == 1:
        ok("empty buffer returns 1 (min clamp)")
    else:
        fail(f"empty buffer returned {mem.get_context_size()}, expected 1")


def test_get_context_isolation():
    print("\nget_context returns copies")
    mem = BufferMemory(max_messages=6)
    mem.add("user", "original")
    ctx = mem.get_context()
    ctx[0]["content"] = "mutated"

    fresh = mem.get_context()
    if fresh[0]["content"] == "original":
        ok("mutating returned context does not affect internal state")
    else:
        fail("get_context leaks internal reference")


def test_chatbot_basic():
    print("\n chatbot integration (requires GEMINI_API_KEY)")
    try:
        from chatbot import Chatbot

        mem = BufferMemory(max_messages=6)
        bot = Chatbot(mem)

        answer = bot.chat("Reply with exactly: PONG")
        if answer and "[ERROR" not in answer:
            ok(f"got non-empty answer: {answer[:60]}")
        else:
            fail(f"unexpected answer: {answer}")

        stats = bot.get_summary_stats()
        if stats["total_turns"] == 1:
            ok("total_turns=1 after one chat()")
        else:
            fail(f"total_turns={stats['total_turns']}, expected 1")

        bot.reset()
        if bot.get_summary_stats()["total_turns"] == 0:
            ok("total_turns=0 after reset()")
        else:
            fail("reset() did not clear turn counter")

    except EnvironmentError as exc:
        print(f"  skipped (no API key): {exc}")
    except Exception as exc:
        fail(f"unexpected error: {exc}")


if __name__ == "__main__":
    print("BufferMemory + Chatbot tests")

    test_overflow()
    test_clear()
    test_validation()
    test_context_size()
    test_get_context_isolation()
    test_chatbot_basic()

    print(f"Results: {_passed} passed, {_failed} failed")

    sys.exit(0 if _failed == 0 else 1)
