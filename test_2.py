# Tests for SummaryMemory

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from google import genai

load_dotenv()

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


def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    return genai.Client(api_key=key)


def test_validation():
    print("\nconstructor validation")
    from memories.summary_memory import SummaryMemory

    try:
        SummaryMemory(max_messages=1)
        fail("should reject max_messages=1")
    except ValueError:
        ok("rejected max_messages=1")

    try:
        SummaryMemory(max_messages=5, summarize_threshold=5)
        fail("should reject threshold >= max_messages")
    except ValueError:
        ok("rejected threshold >= max_messages")

    try:
        SummaryMemory(max_messages=5, summarize_threshold=6)
        fail("should reject threshold > max_messages")
    except ValueError:
        ok("rejected threshold > max_messages")


def test_clear():
    print("\nclear()")
    from memories.summary_memory import SummaryMemory

    mem = SummaryMemory(max_messages=6, llm=None, summarize_threshold=3)
    mem.add("user", "hello")
    mem.add("assistant", "hi")
    mem._summary = "injected summary"

    mem.clear()
    s = mem.get_stats()

    checks = [
        (s["history_size"] == 0,   "history cleared"),
        (s["summary_chars"] == 0,  "summary cleared"),
        (s["summary_calls"] == 0,  "summary_calls reset"),
        (s["total_added"] == 0,    "total_added reset"),
        (mem.get_context() == [],  "get_context() returns [] after clear"),
    ]
    for cond, msg in checks:
        ok(msg) if cond else fail(msg)


def test_fallback_no_llm():
    print("\nfallback truncation (no LLM)")
    from memories.summary_memory import SummaryMemory

    mem = SummaryMemory(max_messages=6, llm=None, summarize_threshold=3)
    for i in range(10):
        mem.add("user", f"message {i}")

    s = mem.get_stats()
    if s["summary_calls"] == 0:
        ok("no API calls without LLM")
    else:
        fail(f"unexpected summary_calls={s['summary_calls']}")

    if s["history_size"] <= 6:
        ok(f"history capped at max_messages (size={s['history_size']})")
    else:
        fail(f"history_size={s['history_size']} exceeds max_messages=6")


def test_context_includes_summary():
    print("\nsummary prepended to context")
    from memories.summary_memory import SummaryMemory

    mem = SummaryMemory(max_messages=6, llm=None, summarize_threshold=3)
    mem.add("user", "hello")
    mem._summary = "User said hello earlier."

    ctx = mem.get_context()
    if ctx and "[Conversation so far]" in ctx[0]["content"]:
        ok("summary appears as first context message")
    else:
        fail(f"summary not in context: {ctx[:1]}")

    if len(ctx) == 2: 
        ok("context length = 1 summary + 1 history")
    else:
        fail(f"unexpected context length: {len(ctx)}")


def test_compression_with_api():
    print("\ncompression (requires API)")
    from memories.summary_memory import SummaryMemory

    client = _client()
    mem = SummaryMemory(max_messages=8, llm=client, summarize_threshold=3)

    messages = [
        ("user",      "My name is Alex"),
        ("assistant", "Nice to meet you, Alex!"),
        ("user",      "I work as a data scientist in Kyiv"),
        ("assistant", "That sounds like an interesting job"),
    ]
    for role, content in messages:
        mem.add(role, content)
        time.sleep(0.5)

    s = mem.get_stats()
    if s["summary_calls"] > 0:
        ok(f"compression triggered (calls={s['summary_calls']})")
    else:
        fail("compression did not trigger")

    if s["summary_chars"] > 0:
        ok(f"summary is non-empty ({s['summary_chars']} chars)")
        print(f"     preview: {mem._summary[:100]}")
    else:
        fail("summary is empty after compression")


def test_retention_with_api():
    print("\nretention across distractors (requires API)")
    from memories.summary_memory import SummaryMemory
    from chatbot import Chatbot

    client = _client()
    mem = SummaryMemory(max_messages=6, llm=client, summarize_threshold=3)
    bot = Chatbot(mem)

    for fact in ["My name is Alex", "I live in Kyiv", "My dog is called Max"]:
        bot.chat(fact)
        time.sleep(1.0)

    for distractor in ["What is quantum computing?", "Tell me about neural networks.", "How does DNS work?"]:
        bot.chat(distractor)
        time.sleep(1.0)

    answer = bot.chat("What is my name?")
    print(f"answer: {answer[:120]}")

    if "alex" in answer.lower():
        ok("bot recalled name from summary")
    else:
        fail(f"name not found in answer: {answer}")


if __name__ == "__main__":
    print("=" * 50)
    print("SummaryMemory tests")
    print("=" * 50)

    test_validation()
    test_clear()
    test_fallback_no_llm()
    test_context_includes_summary()

    if os.getenv("GEMINI_API_KEY"):
        try:
            test_compression_with_api()
            test_retention_with_api()
        except EnvironmentError as exc:
            print(f"\n⚠  API tests skipped: {exc}")
        except Exception as exc:
            fail(f"API test crashed: {exc}")
    else:
        print("\n⚠  GEMINI_API_KEY not set — API tests skipped")

    print(f"\n{'=' * 50}")
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 50)

    sys.exit(0 if _failed == 0 else 1)
