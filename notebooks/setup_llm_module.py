#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SETUP LLM MODULE - Google Generative AI Helper
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv


class LLMHelper:
    """Helper class for Google Generative AI (Gemini)"""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize Google Generative AI helper
        
        Args:
            model: Model name (default: gemini-2.5-flash)
        """
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating content: {e}")


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
