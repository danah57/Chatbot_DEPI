import os

def setup_google_api(api_key: str = None):
    """
    Setup Google Generative AI
    
    Steps:
    1. Go to https://aistudio.google.com/app/apikeys
    2. Create new API key
    3. Copy the key
    4. Run this script with the key
    """
    
    print("\n" + "="*80)
    print(" STEP 4: SETUP GOOGLE GENERATIVE AI")
    print("="*80 + "\n")
    
    if api_key:
        # Save to .env file
        with open('.env', 'w') as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
        print(f"API key saved to .env file")
    else:
        print(" No API key provided")
        print("Usage: python setup_google_llm.py 'your-api-key-here'")
    
    # Test connection
    print("\n Testing connection...")
    try:
        import google.generativeai as genai
        if api_key:
            genai.configure(api_key=api_key)
            print("Google API configured successfully!")
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    import sys
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    setup_google_api("AIzaSyBxTXjg0JZ4MXBcFumwM2ILLilI8bwxy3o")