
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SETUP LLM MODULE - Google Generative AI Helper with Retry Logic
"""

import time
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

try:
    from google import genai
except ImportError:
    # Fallback to google.generativeai if google.genai is not available
    import google.generativeai as genai


class LLMHelper:
    """Wrapper for calling Gemini LLM with retry and fallback"""

    def __init__(self, model="gemini-2.5-flash"):
        """Initialize LLM Helper with Gemini model"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file"
            )
        
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception:
            # Fallback for google.generativeai
            genai.configure(api_key=api_key)
            self.client = genai
        
        self.model = model

    def generate_content(self, prompt: str, retries: int = 3, delay: int = 2) -> str:
        """Call Gemini LLM with automatic retry on server overload
        
        Args:
            prompt: Input prompt text
            retries: Number of retry attempts
            delay: Delay in seconds between retries
            
        Returns:
            Generated text response or fallback message
        """
        for attempt in range(retries):
            try:
                if hasattr(self.client, 'models'):
                    # Using google.genai
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                else:
                    # Using google.generativeai
                    model = genai.GenerativeModel(self.model)
                    response = model.generate_content(prompt)
                
                return response.text
            except Exception as e:
                print(f"⚠️ Gemini attempt {attempt + 1}/{retries} failed: {type(e).__name__}")
                if attempt < retries - 1:
                    time.sleep(delay)
        
        # If all retries fail, return fallback message with raw prompt
        return f"⚠️ Unable to generate enhanced response. Here are the matching programs:\n{prompt}"


if __name__ == "__main__":
    try:
        llm = LLMHelper()
        print("✅ LLMHelper initialized successfully!")
        
        # Test
        test_prompt = "What are the benefits of studying abroad?"
        result = llm.generate_content(test_prompt)
        print(f"✅ Test response:\n{result[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")