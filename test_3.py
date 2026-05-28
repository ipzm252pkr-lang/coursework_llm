# Tests for VectorMemory

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def test_import():
    print("\nimport and BaseMemory contract")
    from memories.vector_memory import VectorMemory
    from memories.base import BaseMemory

    mem = VectorMemory()
    if isinstance(mem, BaseMemory):
        ok("VectorMemory is a BaseMemory subclass")
    else:
        fail("VectorMemory does not inherit BaseMemory")

    for method in ["add", "get_context", "get_context_size", "clear", "get_stats"]:
        if hasattr(mem, method):
            ok(f"has method: {method}")
        else:
            fail(f"missing method: {method}")


def test_validation():
    print("\nconstructor and add() validation")
    from memories.vector_memory import VectorMemory

    for kwargs, label in [
        ({"recent_window": 0}, "recent_window=0"),
        ({"top_k": 0},         "top_k=0"),
    ]:
        try:
            VectorMemory(**kwargs)
            fail(f"should reject {label}")
        except ValueError:
            ok(f"rejected {label}")

    mem = VectorMemory()
    for bad_role in ["admin", "system", "User"]:
        try:
            mem.add(bad_role, "text")
            fail(f"should reject role='{bad_role}'")
        except ValueError:
            ok(f"rejected role='{bad_role}'")

    for bad_content in ["", "   ", "\n"]:
        try:
            mem.add("user", bad_content)
            fail(f"should reject empty content {repr(bad_content)}")
        except ValueError:
            ok(f"rejected empty content {repr(bad_content)}")


def test_add_and_collection():
    print("\n-- add() stores to ChromaDB and updates recent window --")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory(recent_window=3, top_k=4)
    msgs = [
        ("user",      "My name is Alex"),
        ("assistant", "Nice to meet you, Alex!"),
        ("user",      "I work as a data scientist"),
        ("assistant", "That is a great field!"),
        ("user",      "I have a dog named Max"),
    ]
    for role, content in msgs:
        mem.add(role, content)

    s = mem.get_stats()
    checks = [
        (len(mem) == 5,           "5 vectors in ChromaDB"),
        (s["total_added"] == 5,   "total_added=5"),
        (s["recent_size"] == 3,   "recent_size=3 (window respected)"),
        (s["collection_size"] == 5, "collection_size=5"),
    ]
    for cond, msg in checks:
        ok(msg) if cond else fail(msg)

    return mem


def test_semantic_search(mem):
    print("\n-- semantic search returns relevant results --")

    ctx = mem.get_context(query="What is my name?")
    contents = [m["content"] for m in ctx]
    print(f"     query: 'What is my name?'")
    for c in contents:
        print(f"     → {c}")

    if any("Alex" in c or "name" in c.lower() for c in contents):
        ok("found name-related message")
    else:
        fail("no name-related message in context")

    ctx2 = mem.get_context(query="Tell me about my pet")
    contents2 = [m["content"] for m in ctx2]
    if any("Max" in c or "dog" in c.lower() for c in contents2):
        ok("found pet-related message for pet query")
    else:
        fail("no pet-related message for pet query")


def test_deduplication():
    print("\nno duplicate messages in context")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory(recent_window=5, top_k=5)
    mem.add("user", "My name is Alex")
    mem.add("assistant", "Hello Alex!")
    mem.add("user", "I love Python")

    ctx = mem.get_context(query="My name is Alex")
    contents = [m["content"] for m in ctx]

    if len(contents) == len(set(contents)):
        ok(f"no duplicates in {len(contents)}-message context")
    else:
        dupes = [c for c in contents if contents.count(c) > 1]
        fail(f"duplicates found: {dupes}")


def test_chronological_order():
    print("\n-- context is sorted chronologically --")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory(recent_window=3, top_k=4)
    for i, content in enumerate([
        "First: dogs are great",
        "Second: cats are nice",
        "Third: dogs again",
        "Fourth: something else",
        "Fifth: dogs one more time",
    ]):
        mem.add("user", content)

    ctx = mem.get_context(query="Tell me about dogs")
    contents = [m["content"] for m in ctx]

    idxs = {label: next((i for i, c in enumerate(contents) if label in c), -1)
            for label in ("First", "Third")}

    if idxs["First"] != -1 and idxs["Third"] != -1:
        if idxs["First"] < idxs["Third"]:
            ok("'First' appears before 'Third' in context")
        else:
            fail(f"wrong order: 'First' at {idxs['First']}, 'Third' at {idxs['Third']}")
    else:
        ok(f"chronological order preserved ({len(ctx)} messages returned)")


def test_clear():
    print("\nclear() resets all state")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory()
    mem.add("user", "hello")
    mem.add("assistant", "hi")

    mem.clear()
    s = mem.get_stats()

    checks = [
        (len(mem) == 0,             "ChromaDB collection empty"),
        (s["recent_size"] == 0,     "recent window cleared"),
        (s["total_added"] == 0,     "total_added reset"),
        (mem.get_context() == [],   "get_context() returns []"),
        (s["retrieval"]["count"] == 0, "retrieval stats reset"),
    ]
    for cond, msg in checks:
        ok(msg) if cond else fail(msg)


def test_retrieval_stats():
    print("\n_retrieval_stats() tracks cosine similarities")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory()
    stats = mem._retrieval_stats()

    checks = [
        (stats["avg"] == 0.0,   "avg=0.0 before any search"),
        (stats["count"] == 0,   "count=0 before any search"),
    ]
    for cond, msg in checks:
        ok(msg) if cond else fail(msg)

    mem.add("user", "My name is Alex")
    mem.add("assistant", "Hello Alex")
    mem.get_context(query="What is my name?")

    stats = mem._retrieval_stats()
    if 0.0 <= stats["avg"] <= 1.0:
        ok(f"avg similarity in [0,1]: {stats['avg']}")
    else:
        fail(f"avg out of range: {stats['avg']}")

    if stats["count"] > 0:
        ok(f"retrieval count > 0: {stats['count']}")
    else:
        fail("count still 0 after search")


def test_no_query_returns_recent():
    print("\nget_context(query=None) returns recent window only")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory(recent_window=3)
    for i in range(5):
        mem.add("user", f"message {i}")

    ctx = mem.get_context(query=None)
    if len(ctx) == 3:
        ok("returns exactly recent_window=3 messages")
    else:
        fail(f"expected 3, got {len(ctx)}")

    if ctx[-1]["content"] == "message 4":
        ok("last message is the most recent")
    else:
        fail(f"unexpected last message: {ctx[-1]['content']}")


def test_empty_collection():
    print("\nget_context on empty collection")
    from memories.vector_memory import VectorMemory

    mem = VectorMemory()
    ctx_with_query = mem.get_context(query="anything")
    ctx_without = mem.get_context(query=None)

    if ctx_with_query == []:
        ok("get_context(query=...) on empty returns []")
    else:
        fail(f"expected [], got {ctx_with_query}")

    if ctx_without == []:
        ok("get_context(query=None) on empty returns []")
    else:
        fail(f"expected [], got {ctx_without}")


def test_chatbot_integration():
    print("\nchatbot integration (requires GEMINI_API_KEY)")
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        print("skipped (no API key)")
        return

    try:
        from memories.vector_memory import VectorMemory
        from chatbot import Chatbot

        mem = VectorMemory(recent_window=3, top_k=4)
        bot = Chatbot(mem)

        bot.chat("My name is Alex and I love Python")
        time.sleep(1.5)
        bot.chat("What is machine learning?")
        time.sleep(1.5)

        answer = bot.chat("What is my name?")
        print(f"     answer: {answer[:120]}")

        if "alex" in answer.lower():
            ok("recalled name via vector search")
        else:
            fail(f"name not found in answer: {answer}")

    except Exception as exc:
        fail(f"integration test crashed: {exc}")


if __name__ == "__main__":
    print("VectorMemory tests")

    try:
        test_import()
        test_validation()
        mem = test_add_and_collection()
        test_semantic_search(mem)
        test_deduplication()
        test_chronological_order()
        test_clear()
        test_retrieval_stats()
        test_no_query_returns_recent()
        test_empty_collection()
        test_chatbot_integration()

    except ImportError as exc:
        print(f"\nImport failed: {exc}")
        print("Install deps: pip install chromadb sentence-transformers")
        sys.exit(1)

    print(f"Results: {_passed} passed, {_failed} failed")

    sys.exit(0 if _failed == 0 else 1)