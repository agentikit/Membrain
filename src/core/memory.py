# agent_memory_search.py

import os
import math
import json
import tokenize
from io import BytesIO
from typing import Dict, List, Any

class AgentMemorySearch:
    def __init__(self):
        self.inverted_index = {}   # term -> {doc_id: freq}
        self.doc_store = {}        # doc_id -> JSON object
        self.doc_lengths = {}      # doc_id -> token count
        self.schema = []           # list of fields to index
        self.doc_id_counter = 1

    def set_schema(self, fields: List[str]):
        self.schema = fields

    def tokenize_text(self, text: str) -> List[str]:
        try:
            tokens = []
            for tok in tokenize.tokenize(BytesIO(text.encode('utf-8')).readline):
                if tok.type in (tokenize.NAME, tokenize.STRING, tokenize.COMMENT):
                    tokens.append(tok.string.lower())
        except tokenize.TokenError:
            tokens = text.lower().split()
        return tokens

    def index_json(self, data: Dict[str, Any]):
        doc_id = self.doc_id_counter
        self.doc_id_counter += 1

        tokens = []
        for field in self.schema:
            if field in data and isinstance(data[field], str):
                tokens.extend(self.tokenize_text(data[field]))

        self.doc_store[doc_id] = data
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            self.inverted_index.setdefault(token, {}).setdefault(doc_id, 0)
            self.inverted_index[token][doc_id] += 1

    def compute_idf(self, term: str) -> float:
        df = len(self.inverted_index.get(term, {}))
        total_docs = len(self.doc_lengths)
        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, query_terms: List[str], doc_id: int, k=1.5, b=0.75) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        for term in query_terms:
            tf = self.inverted_index.get(term, {}).get(doc_id, 0)
            idf = self.compute_idf(term)
            denom = tf + k * (1 - b + b * doc_len / avg_doc_len)
            score += idf * ((tf * (k + 1)) / (denom + 1e-10))
        return score

    def search(self, query: str, top_k=5) -> List[Dict[str, Any]]:
        query_terms = self.tokenize_text(query)
        scores = {}
        for doc_id in self.doc_store:
            score = self.bm25_score(query_terms, doc_id)
            if score > 0:
                scores[doc_id] = score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": doc_id, "data": self.doc_store[doc_id], "score": score} for doc_id, score in sorted_results[:top_k]]