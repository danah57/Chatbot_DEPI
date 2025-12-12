🎓 Study Abroad Helper Chatbot

An AI-powered conversational system that helps students explore and compare international universities using a Retrieval-Augmented Generation (RAG) pipeline.

📌 Table of Contents

*   [About the Project](#-about-the-project)
Key Features
Demo
Prerequisites
Installation
Quick Start
Usage Examples
Project Structure
Technologies Used
How It Works
API Reference
Testing
Deployment
Roadmap
Contributing
License
Contact & Support


📖 About the Project
This project is a complete AI-powered system that helps students explore and compare international universities simply ask questions in natural language and get intelligent, data-driven responses.
The chatbot uses:

Natural Language Processing (NLP) to understand student queries
LangChain to manage multi-turn conversations with memory
OpenAI GPT to generate human-like responses
Pandas & Python to efficiently query and filter university data
Streamlit to create an intuitive web interface

Project Objectives:
✅ Develop NLP-based chatbot for university information retrieval
✅ Implement data handling for efficient querying
✅ Create interactive UI for seamless user experience
✅ Ensure response accuracy and relevance
✅ Enable multi-turn conversations with context retention

## Recent Work & Integration Summary

This repository was extended with a full Retrieval-Augmented Generation (RAG) pipeline and production-ready integrations. Changes and additions made in this workspace include:

- **RAG pipeline and embeddings**: a notebook/script that builds text descriptions from the program dataset and creates SentenceTransformer embeddings (model: `all-MiniLM-L6-v2`). Embeddings are saved as `embeddings.npy` and a compatibility `embeddings.pkl`, and metadata is saved as `metadata.csv` in `./data/processed/`.

- **FAISS index support**: optional FAISS index creation (`IndexFlatIP`) for fast cosine-similarity retrieval from normalized vectors. Index files are stored under `./data/processed/` (e.g. `faiss_index.bin`).

- **LLM integration with retry & fallback**: created `notebooks/setup_llm2.py` which contains `LLMHelper` — wraps the Google Generative AI (Gemini) client, supports both `GEMINI_API_KEY` and `GOOGLE_API_KEY`, implements retry logic (3 attempts) and returns a safe fallback if the API is unavailable.

- **Streamlit UI integration**: a dedicated app (`newapp.py`) now dynamically loads the RAG system and exposes a chat UI. The RAG system is cached in Streamlit session state and returns both the generated answer and the matched program records.

- **Robust environment loading**: `.env` loading is handled from multiple candidate locations; the LLM helper checks both `GEMINI_API_KEY` and `GOOGLE_API_KEY` environment variables.

- **Improved developer ergonomics**: the Streamlit app prints the full RAG result (JSON-friendly) to the console for debugging; logging and debug modes are available.

- **Compatibility & dependencies**: updated `requirements.txt` contains `sentence-transformers`, `faiss-cpu`, `google-generativeai` (or compatible `google.genai`), and other required libraries. Installation with `pip install -r requirements.txt` is recommended before running the app.

Where to look in the repo:

- `notebooks/02_NLP_and_Embeddings.py` — create embeddings, metadata, optional Faiss index
- `notebooks/setup_llm2.py` — `LLMHelper` with retry/fallback
- `notebooks/05_rag_system.py` — RAG system orchestration (encode, search, format, prompt)
- `newapp.py` — Streamlit UI integrated with RAG system
- `.env` / `.env.example` — API key and config

How to run the integrated system (quick):

1. Create and activate virtualenv, then install dependencies:

```powershell
pip install -r requirements.txt
```

2. Ensure your `.env` contains the LLM API key (examples accept `GEMINI_API_KEY` or `GOOGLE_API_KEY`):

```dotenv
GEMINI_API_KEY=your_key_here
```

3. (Optional) Recreate embeddings and FAISS index if you change data:

```powershell
python notebooks/02_NLP_and_Embeddings.py ./data/all_programs_cleaned.xlsx --output-dir ./data/processed --faiss
python notebooks/03_faiss_index.py ./data/processed/embeddings.pkl --output-dir ./data/processed
```

4. Start the Streamlit app:

```powershell
streamlit run newapp.py
```

Open `http://localhost:8501` and test queries (the full RAG result will be printed in the terminal for debugging).

⭐ Key Features
🤖 Smart Conversational AI

Understands natural language queries about universities
Maintains conversation context across multiple messages
Provides accurate, relevant information from database

🔍 Advanced Search & Filtering

Search by university name, country, program
Filter by budget, tuition fees, language requirements
Filter by admission criteria (GPA, IELTS, TOEFL scores)
Find programs by degree level (Bachelor, Masters, PhD)

📊 Data Exploration

Browse all universities and programs
View detailed program information
Compare multiple programs side-by-side
See statistics about database

💾 Flexible Data Management

Upload custom university datasets (CSV format)
Automatic data cleaning and preprocessing
Support for real-world data variations

🌍 Comprehensive University Data

60+ universities across 10+ countries
Multiple programs per university
Tuition fees, duration, language requirements
Admission criteria and program descriptions


🎬 Demo
Try the Live Demo:
bashstreamlit run app.py
Sample Conversations:
Student: "Universities in USA offering Computer Science"
Bot: Shows Harvard, MIT, Stanford CS programs with details
Student: "Masters programs under $50,000"
Bot: Filters and displays affordable programs worldwide
Student: "What are IELTS requirements for Harvard?"
Bot: Retrieves and displays admission requirements

📋 Prerequisites
Before you begin, ensure you have:

Python 3.8 or higher - Download from python.org
pip - Python package manager (included with Python)
Git - Download from git-scm.com
OpenAI API Key - Get from platform.openai.com
Text Editor - VS Code, PyCharm, or any editor
2GB RAM minimum - For running the application

Check Installation:
bashpython --version
pip --version
git --version

🚀 Installation
Step 1: Clone Repository
bashgit clone https://github.com/YOUR_USERNAME/university-admission-chatbot.git
cd university-admission-chatbot
Step 2: Create Virtual Environment
On macOS/Linux:
bashpython3 -m venv venv
source venv/bin/activate
On Windows:
bashpython -m venv venv
venv\Scripts\activate
You should see (venv) in your terminal.
Step 3: Install Dependencies
bashpip install -r requirements.txt
This installs:

streamlit - Web UI framework
langchain - LLM orchestration
openai - GPT API access
transformers - NLP models
pandas - Data processing
torch - Deep learning library

Step 4: Setup Environment Variables
bash# Copy the example file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# Open .env in your text editor and replace:
# OPENAI_API_KEY=your_actual_api_key_here
Getting OpenAI API Key:

Go to https://platform.openai.com/account/api-keys
Click "Create new secret key"
Copy the key
Paste into .env file

Step 5: Preprocess Dataset
bashpython process_data.py
You should see: ✅ Data processing complete!
Step 6: Run the Application
bashstreamlit run app.py
Your browser will automatically open to http://localhost:8501

📝 Quick Start
Running for First Time
bash# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run the app
streamlit run app.py
Deactivating Virtual Environment
bashdeactivate
Stopping the App
Press Ctrl + C in terminal

💬 Usage Examples
Example 1: Search by Country
You: "Show me all universities in Canada"
Bot: Displays all Canadian universities with their programs and tuition fees
Example 2: Budget Filter
You: "I have a budget of $40,000, what programs can I afford?"
Bot: Shows all programs costing less than $40,000 per year
Example 3: Admission Requirements
You: "What's the minimum IELTS score for MBA programs?"
Bot: Retrieves and displays IELTS requirements for all MBA programs
Example 4: Program Comparison
You: "Compare Computer Science programs at MIT and Stanford"
Bot: Shows side-by-side comparison with duration, fees, requirements
Example 5: Degree Level Filter
You: "Find all Masters programs in the USA"
Bot: Lists all Masters degrees available in United States universities
Example 6: Multi-turn Conversation
You: "Tell me about universities in UK"
Bot: Shows British universities
You: "Which one has the lowest tuition?"
Bot: Compares tuition and recommends the most affordable

📁 Project Structure
study_abroad_chatbot/
│
├── 📂 data/
│   ├── raw/
│   │   └── all_programs_cleaned.xlsx        # Original raw data
│   └── processed/
│       ├── universities_data.csv            # Cleaned and preprocessed data
│       ├── embeddings.pkl                    # Embedding vectors
│       └── faiss_index.bin                   # FAISS index
│
├── 📂 scripts/
│   ├── 01_data_loading.py                   # Load and clean data
│   ├── 02_create_embeddings.py              # Generate embeddings
│   ├── 03_build_faiss_index.py              # Build FAISS index
│   ├── 04_setup_google_llm.py               # Setup Google LLM API
│   ├── 05_rag_system.py                     # RAG system implementation
│   └── 06_test_system.py                    # Test the RAG system
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_processor.py                    # Data preprocessing functions
│   ├── embeddings_handler.py                # Embeddings management
│   ├── faiss_manager.py                     # FAISS index management
│   └── rag_pipeline.py                      # RAG pipeline and retrieval logic
│
├── 📄 app.py                                # Main Streamlit app
├── 📄 .env                                  # API keys and environment variables
├── 📄 requirements.txt                      # Required Python libraries
├── 📄 README.md                             # Documentation
└── 📄 .gitignore  
File Descriptions:
FilePurposeapp.pyMain Streamlit web app - runs the chatbot UIsrc/data_loader.pyLoads CSV and preprocesses university datasrc/query_processor.pyExtracts entities and classifies intent from user queriessrc/data_query_handler.pySearches and filters the databasesrc/chatbot_engine.pyOrchestrates all components and manages conversationsrequirements.txtLists all Python package dependencies.envStores sensitive information (API keys)tests/Contains pytest unit tests for all modulesdocs/Complete documentation and guides

🛠️ Technologies Used
Backend

Python 3.8+ - Core programming language
LangChain - Manages conversation flow and memory
OpenAI API - GPT-3.5/GPT-4 for response generation
Transformers (Hugging Face) - BERT for NLP tasks
PyTorch - Deep learning framework

Data Processing

Pandas - Data manipulation and analysis
NumPy - Numerical computing
Scikit-learn - Machine learning utilities

Frontend

Streamlit - Web UI framework (no HTML/CSS needed)
Plotly - Data visualization

Development

Pytest - Unit testing framework
Git - Version control
GitHub - Repository hosting

Infrastructure

Python venv - Virtual environment
pip - Package manager


🧠 How It Works
Architecture Flow
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│                    (Streamlit - app.py)                      │
│                                                              │
│  • Chat input box                                            │
│  • Display messages & data                                   │
│  • Data exploration panels                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              CHATBOT ENGINE (chatbot_engine.py)              │
│                                                              │
│  • Orchestrates components                                   │
│  • Maintains conversation memory (LangChain)                │
│  • Generates responses using OpenAI                         │
└─────────────────────────────────────────────────────────────┘
                   ↙          ↓          ↖
       ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
       │   QUERY      │  │     RAG      │  │  RESPONSE    │
       │  PROCESSOR   │  │    SYSTEM    │  │ GENERATION   │
       │              │  │              │  │              │
       │ • Extract    │  │ • Filter by  │  │ • Use LLM    │
       │   entities   │  │   criteria   │  │ • Format     │
       │ • Classify   │  │ • Search DB  │  │   answer     │
       │   intent     │  │ • Aggregate  │  │ • Add context│
       └──────────────┘  └──────────────┘  └──────────────┘
                            ↓
       ┌─────────────────────────────────────┐
       │     UNIVERSITY DATABASE              │
       │   (data/processed/universities.csv) │
       │                                      │
       │  • 60+ universities                 │
       │  • 200+ programs                    │
       │  • Complete details                 │
       └─────────────────────────────────────┘
Processing Pipeline

User Input → Student types a question
Query Processing → Extract entities (university, country, budget, etc.)
Intent Classification → Understand what student is asking (search, filter, compare)
Data Retrieval → Query database based on extracted entities
Response Generation → Use LLM to format data into natural language
Memory Management → Store conversation for context in next message
Display → Show response and retrieved data to user


🔌 API Reference
Main Modules
UniversityDataLoader (data_loader.py)
pythonloader = UniversityDataLoader("path/to/data.csv")
loader.preprocess_data()
universities = loader.get_universities()
programs = loader.get_programs_by_university("MIT")
QueryProcessor (query_processor.py)
pythonprocessor = QueryProcessor()
entities = processor.extract_entities("Masters in USA under 50000")
intent = processor.classify_intent("Compare programs")
UniversityQueryHandler (data_query_handler.py)
pythonhandler = UniversityQueryHandler(dataframe)
usa_unis = handler.search_by_country("USA")
cs_programs = handler.search_by_program("Computer Science")
affordable = handler.filter_by_budget(40000)
UniversityChatbot (chatbot_engine.py)
pythonchatbot = UniversityChatbot(dataframe)
response, context = chatbot.process_query("Find USA universities")
See API_DOCUMENTATION.md for detailed specs.

✅ Testing
Run All Tests
bashpytest tests/ -v
Run Specific Test
bashpytest tests/test_chatbot.py -v
Run with Coverage
bashpytest tests/ --cov=src
Test Results
All tests should pass with ✅ marks:

✅ test_data_loading
✅ test_chatbot_initialization
✅ test_query_processing
✅ test_entity_extraction


🌐 Deployment
Option 1: Streamlit Cloud (Recommended)

Push code to GitHub
Go to share.streamlit.io
Connect GitHub repository
Select app.py as main file
Add secrets (OpenAI API key) in settings
Deploy!

Option 2: Heroku
bash# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Push to Heroku
git push heroku main
Option 3: Docker
bashdocker build -t university-chatbot .
docker run -p 8501:8501 university-chatbot
See DEPLOYMENT.md for detailed instructions.

🗺️ Roadmap
✅ Completed

Basic chatbot architecture
Query processing and NLP
Streamlit UI
Data filtering and search
Multi-turn conversations

🔄 In Progress

RAG (Retrieval-Augmented Generation) implementation
Vector database integration (FAISS)
Advanced response ranking

⏳ Planned

Scholarship information
Application tracking
User accounts and preferences
Mobile app version
Multi-language support
Voice input/output
Analytics dashboard


🤝 Contributing
Contributions are welcome! Follow these steps:

Fork the repository

bash   git clone https://github.com/YOUR_USERNAME/university-admission-chatbot.git

Create a feature branch

bash   git checkout -b feature/AmazingFeature

Make your changes and commit

bash   git commit -m "Add: AmazingFeature description"

Push to branch

bash   git push origin feature/AmazingFeature

Open a Pull Request on GitHub

Contribution Guidelines

Follow PEP 8 Python style guide
Add docstrings to all functions
Write unit tests for new features
Update documentation
Keep commits atomic and well-described


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...

📞 Contact & Support
Get Help

📖 Check Troubleshooting Guide
💬 Open an Issue
📧 Email: your.email@example.com

Questions?

How do I add more universities?
How do I change the model?
How do I customize the UI?

See USAGE_GUIDE.md for FAQs.
Report Bugs
Found a bug? Please create an issue with:

Description of the problem
Steps to reproduce
Expected vs actual behavior
Error messages/screenshots

Suggest Features
Have an idea? Open an issue with feature label and describe your suggestion.

🙏 Acknowledgments

OpenAI for GPT API
Hugging Face for Transformers
Streamlit for the amazing framework
LangChain for conversation management
All contributors and users


📊 Project Statistics
MetricValueLines of Code2,000+Modules5Functions30+Test Cases10+Documentation Pages6Universities60+Programs200+

🚀 Quick Links

Live Demo
Setup Guide
API Docs
Usage Examples
Architecture


<div align="center">

⭐ If you found this helpful, please star the repository!
</div>
