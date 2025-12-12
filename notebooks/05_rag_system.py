#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT 5: RAG SYSTEM WITH GEMINI LLM HELPER
FINAL WORKING VERSION - Fully Integrated with LLMHelper
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from multiple possible locations
env_paths = [
    Path(__file__).parent.parent / ".env",  # From notebooks/..
    Path.cwd() / ".env",  # From current working directory
    Path(".env"),  # Relative to current dir
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        break

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import Dict
from langchain_core.prompts import PromptTemplate
from setup_llm2 import LLMHelper


class RAGChatbot:
    """Complete RAG Chatbot with Gemini LLM Helper"""

    def __init__(self, data_path: str, embeddings_path: str, index_path: str):
        """Initialize RAG system"""
        print("\n" + "="*80)
        print("🤖 INITIALIZING RAG SYSTEM")
        print("="*80 + "\n")

        # --- Load data ---
        print("📚 Loading data...")
        try:
            self.data = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(data_path, encoding='latin-1')
            except:
                print("⚠️ CSV encoding issue, trying Excel...")
                self.data = pd.read_excel(data_path.replace('.csv', '.xlsx'))
        print(f"✅ Data loaded: {len(self.data)} records")

        # --- Load embeddings ---
        print("📊 Loading embeddings...")
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        print(f"✅ Embeddings loaded: shape {self.embeddings.shape}")

        # --- Load FAISS index ---
        print("⚡ Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print(f"✅ Index loaded: {self.index.ntotal} vectors")

        # --- Load embedding model ---
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("✅ Embedding model loaded!")

        # --- Initialize Gemini LLM Helper ---
        print("🌐 Initializing Gemini LLM Helper...")
        try:
            self.llm_helper = LLMHelper(model="gemini-2.5-flash")
            print("✅ Gemini LLM Helper initialized!")
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini LLM Helper: {e}")
            self.llm_helper = None

        # --- Prompt templates ---
        self.prompt_templates = self._create_prompt_templates()
        self.history = []
        print("\n✅ RAG System ready!\n")

    # ---------------- Prompt Templates ----------------
    def _create_prompt_templates(self) -> Dict:
        templates = {
            'search': PromptTemplate(
                input_variables=['query', 'programs'],
                template="""You are a helpful university advisor.

User Query: {query}

Found Programs:
{programs}

Provide a helpful response recommending the best options."""
            ),
            'comparison': PromptTemplate(
                input_variables=['query', 'programs'],
                template="""You are a university advisor specializing in program comparison.

User Query: {query}

Programs to Compare:
{programs}

Provide a detailed comparison with pros and cons."""
            ),
            'recommendation': PromptTemplate(
                input_variables=['query', 'programs'],
                template="""You are an expert university advisor.

User Query: {query}

Available Programs:
{programs}

Recommend the best options with reasoning."""
            )
        }
        return templates

    # ---------------- Intent Classification ----------------
    def _classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['compare', 'vs', 'difference', 'between']):
            return 'comparison'
        elif any(word in query_lower for word in ['recommend', 'best', 'should', 'suggest']):
            return 'recommendation'
        else:
            return 'search'

    # ---------------- Safe value extraction ----------------
    def _safe_get_value(self, value):
        try:
            if pd.isna(value):
                return None
            return value
        except:
            return None

    # ---------------- Format programs ----------------
    def _format_programs(self, indices: np.ndarray, distances: np.ndarray) -> str:
        formatted_list = []
        if len(indices.shape) > 1:
            indices = indices[0]
            distances = distances[0]
        for i, idx in enumerate(indices):
            idx_int = int(idx)
            row = self.data.iloc[idx_int]
            similarity = 1 / (1 + float(distances[i]))

            program = str(self._safe_get_value(row.get('program', 'N/A'))).strip()
            university = str(self._safe_get_value(row.get('university_name', 'N/A'))).strip()
            duration = str(self._safe_get_value(row.get('duration', 'N/A'))).strip()

            fees_val = self._safe_get_value(row.get('fees', 0))
            try:
                fees = float(fees_val) if fees_val else 0
                fees_str = f"${fees:,.0f}" if fees > 0 else "N/A"
            except:
                fees_str = "N/A"

            info = f"{i+1}. {program}\n"
            info += f"   University: {university}\n"
            info += f"   Fees: {fees_str}\n"
            info += f"   Duration: {duration}\n"

            ielts_val = self._safe_get_value(row.get('ielts', 0))
            try:
                ielts = float(ielts_val) if ielts_val else 0
                if ielts > 0:
                    info += f"   IELTS: {ielts}\n"
            except:
                pass

            toefl_val = self._safe_get_value(row.get('toefl', 0))
            try:
                toefl = float(toefl_val) if toefl_val else 0
                if toefl > 0:
                    info += f"   TOEFL: {toefl}\n"
            except:
                pass

            info += f"   Match: {similarity:.2%}\n"
            formatted_list.append(info)

        return "\n".join(formatted_list)

    # ---------------- Answer Query ----------------
    def answer(self, query: str, k: int = 5) -> Dict:
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            query_f32 = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_f32, k)

            intent = self._classify_intent(query)
            programs_text = self._format_programs(indices, distances)
            prompt_template = self.prompt_templates.get(intent, self.prompt_templates['search'])
            prompt_text = prompt_template.format(query=query, programs=programs_text)

            # ---------------- Use Gemini LLM ----------------
            if self.llm_helper:
                try:
                    response_text = self.llm_helper.generate_content(prompt_text)
                except Exception as e:
                    print(f"⚠️ Gemini LLM error: {e}")
                    response_text = f"Found {len(indices[0])} programs:\n\n{programs_text}"
            else:
                response_text = f"Found {len(indices[0])} programs:\n\n{programs_text}"

            self.history.append({
                'query': query,
                'intent': intent,
                'response': response_text,
                'results': self.data.iloc[indices[0]] if len(indices.shape) > 1 else self.data.iloc[indices],
                'distances': distances
            })

            return {
                'response': response_text,
                'programs': self.data.iloc[indices[0]] if len(indices.shape) > 1 else self.data.iloc[indices],
                'intent': intent,
                'count': len(indices[0]) if len(indices.shape) > 1 else len(indices),
                'indices': indices,
                'distances': distances
            }

        except Exception as e:
            print(f"❌ Error in answer(): {e}")
            import traceback
            traceback.print_exc()
            return {
                'response': f"Error processing query: {str(e)}",
                'programs': None,
                'intent': 'error',
                'count': 0
            }

# ---------------- TEST ----------------
if __name__ == "__main__":
    chatbot = RAGChatbot(
        data_path='./data/processed/universities_data.csv',
        embeddings_path='./data/processed/embeddings.pkl',
        index_path='./data/processed/faiss_index.bin'
    )

    test_queries = [
        "Find cheap engineering programs",
        "Compare master's programs",
        "Recommend best options"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = chatbot.answer(query, k=3)
        print(f"✅ Intent: {result['intent']}")
        print(f"✅ Found: {result['count']} programs\n")
        print("Response:")
        print(result['response'][:400])
        if len(result['response']) > 400:
            print("...")
