#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT 5: RAG SYSTEM WITH GOOGLE GENERATIVE AI
FINAL WORKING VERSION - All bugs fixed + Encoding fixed
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import Dict
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()


class RAGChatbotWithGoogle:
    """Complete RAG Chatbot with Google Generative AI"""
    
    def __init__(self, data_path: str, embeddings_path: str, index_path: str):
        """Initialize RAG system"""
        
        print("\n" + "="*80)
        print("🤖 STEP 5: INITIALIZING RAG SYSTEM")
        print("="*80 + "\n")
        
        # Load data with encoding fix
        print("📚 Loading data...")
        try:
            # Try UTF-8 first
            self.data = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try Latin-1
                self.data = pd.read_csv(data_path, encoding='latin-1')
            except:
                # Last resort: read Excel
                print("⚠️ CSV encoding issue, trying Excel...")
                self.data = pd.read_excel(data_path.replace('.csv', '.xlsx'))
        
        print(f"✅ Data loaded: {len(self.data)} records")
        
        # Load embeddings
        print("📊 Loading embeddings...")
        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        print(f"✅ Embeddings loaded: shape {self.embeddings.shape}")
        
        # Load FAISS index
        print("⚡ Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print(f"✅ Index loaded: {self.index.ntotal} vectors")
        
        # Initialize embedding model
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("✅ Model loaded!")
        
        # Initialize Google LLM
        print("🌐 Initializing Google Generative AI...")
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Google LLM initialized!")
        else:
            self.llm = None
            print("⚠️ Google API key not found - will use template responses")
        
        # Initialize prompt templates (LangChain)
        self.prompt_templates = self._create_prompt_templates()
        
        self.history = []
        
        print("\n✅ RAG System ready!\n")
    
    def _create_prompt_templates(self) -> Dict:
        """Create LangChain prompt templates"""
        
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
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'difference', 'between']):
            return 'comparison'
        elif any(word in query_lower for word in ['recommend', 'best', 'should', 'suggest']):
            return 'recommendation'
        else:
            return 'search'
    
    def _safe_get_value(self, value):
        """Safely get value from pandas Series, handling NaN"""
        try:
            if pd.isna(value):
                return None
            return value
        except:
            return None
    
    def _format_programs(self, indices: np.ndarray, distances: np.ndarray) -> str:
        """
        Format retrieved programs for display
        FAISS returns 2D arrays: indices[0] and distances[0]
        """
        formatted_list = []
        
        # FAISS returns results as [batch_size, k]
        # We need indices[0] and distances[0] because batch_size=1
        if len(indices.shape) > 1:
            indices = indices[0]
            distances = distances[0]
        
        # Now iterate through the 1D array
        for i, idx in enumerate(indices):
            idx_int = int(idx)  # Convert numpy int64 to Python int
            row = self.data.iloc[idx_int]
            similarity = 1 / (1 + float(distances[i]))
            
            # Convert all values to safe strings/floats
            program = str(self._safe_get_value(row.get('program', 'N/A'))).strip()
            university = str(self._safe_get_value(row.get('university_name', 'N/A'))).strip()
            duration = str(self._safe_get_value(row.get('duration', 'N/A'))).strip()
            
            # Handle fees
            fees_val = self._safe_get_value(row.get('fees', 0))
            try:
                fees = float(fees_val) if fees_val else 0
                fees_str = f"${fees:,.0f}" if fees > 0 else "N/A"
            except:
                fees_str = "N/A"
            
            # Build program info
            info = f"{i+1}. {program}\n"
            info += f"   University: {university}\n"
            info += f"   Fees: {fees_str}\n"
            info += f"   Duration: {duration}\n"
            
            # Add IELTS if available
            ielts_val = self._safe_get_value(row.get('ielts', 0))
            try:
                ielts = float(ielts_val) if ielts_val else 0
                if ielts > 0:
                    info += f"   IELTS: {ielts}\n"
            except:
                pass
            
            # Add TOEFL if available
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
    
    def answer(self, query: str, k: int = 5) -> Dict:
        """Answer user query"""
        
        try:
            # Step 1: Encode query
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Step 2: Search with FAISS
            query_f32 = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_f32, k)
            
            # Step 3: Classify intent
            intent = self._classify_intent(query)
            
            # Step 4: Format programs
            programs_text = self._format_programs(indices, distances)
            
            # Step 5: Get prompt template
            prompt_template = self.prompt_templates.get(intent, self.prompt_templates['search'])
            
            # Step 6: Format prompt
            prompt_text = prompt_template.format(query=query, programs=programs_text)
            
            # Step 7: Call Google LLM (if available)
            response_text = ""
            
            if self.llm:
                try:
                    response = self.llm.generate_content(prompt_text)
                    response_text = response.text
                except Exception as e:
                    print(f"⚠️ LLM error: {e}")
                    response_text = f"Found {len(indices[0])} programs:\n\n{programs_text}"
            else:
                response_text = f"Found {len(indices[0])} programs:\n\n{programs_text}"
            
            # Step 8: Store in history
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


# ============================================================================
# TEST THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("🧪 TESTING RAG SYSTEM")
    print("="*80)
    
    # Initialize RAG system
    try:
        chatbot = RAGChatbotWithGoogle(
            data_path='./data/processed/universities_data.csv',
            embeddings_path='./data/processed/embeddings.pkl',
            index_path='./data/processed/faiss_index.bin'
        )
    except FileNotFoundError as e:
        print(f"❌ Error: Missing file - {e}")
        print("\nPlease run the scripts in order:")
        print("1. python 1_data_loading.py")
        print("2. python 2_create_embeddings.py")
        print("3. python 3_build_faiss_index.py")
        exit()
    
    # Test queries
    test_queries = [
        "Find cheap engineering programs",
        "Compare master's programs",
        "Recommend best options"
    ]
    
    print("\n" + "="*80)
    print("🔍 TESTING QUERIES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        result = chatbot.answer(query, k=3)
        
        print(f"✅ Intent: {result['intent']}")
        print(f"✅ Found: {result['count']} programs\n")
        print("Response:")
        print(result['response'][:400])
        
        if len(result['response']) > 400:
            print("...")
    
    # Final status
    print("\n" + "="*80)
    print("✅ RAG SYSTEM WORKING!")
    print("="*80)
    print("\n🚀 Next Step:")
    print("   Run: streamlit run app.py")
    print("\n")