
import json

from src.core.memory import AgentMemorySearch


def run_test():
    engine = AgentMemorySearch()
    engine.set_schema(["title", "description", "content"])

    sample_data = [
        {
            "title": "How to build a chatbot",
            "description": "Using transformers",
            "content": "This guide explains building a chatbot using HuggingFace models."
        },
        {
            "title": "Memory architecture",
            "description": "Deep learning models",
            "content": "This document discusses RNNs, LSTMs and attention."
        },
        {
            "title": "Build fast search systems",
            "description": "BM25 scoring algorithm",
            "content": "Learn how to implement efficient search using inverted index and BM25."
        },
        {
            "title": "Agent memory design",
            "description": "Schema and indexing for memory",
            "content": "This talks about how agent memory systems use structured JSON schemas."
        }
    ]

    for doc in sample_data:
        engine.index_json(doc)

    test_queries = [
        "chatbot using huggingface",
        "bm25 inverted index",
        "agent memory schema",
        "deep learning attention"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = engine.search(query)
        if not results:
            print("No results found.")
        for result in results:
            print(f"\nDoc ID: {result['doc_id']}, Score: {result['score']:.4f}")
            print(json.dumps(result['data'], indent=2))

if __name__ == "__main__":
    run_test()