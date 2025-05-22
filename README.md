# 🚀 RAG-Based Document Q&A System  

## 📌 Overview  
This project is an **AI-powered Retrieval-Augmented Generation (RAG) system** that allows users to upload documents and ask questions. The system retrieves relevant information from the document and generates **context-aware responses** using an **LLM (DeepSeek R1 via OpenRouter API).**

## 🔍 Features  
✅ Upload documents (PDFs, text files, etc.)  
✅ Automatic text processing & chunking for better retrieval  
✅ Convert text into vector embeddings using **BGE-large**  
✅ Perform **semantic search** using **FAISS vector database**  
✅ Generate **relevant, document-aware responses** using **DeepSeek R1**  

## 🛠️ Tech Stack  
- **Python** – Core backend logic  
- **FAISS** – Fast Approximate Nearest Neighbor (ANN) search  
- **BGE-large embeddings** – High-quality text embeddings  
- **DeepSeek R1 (OpenRouter API)** – LLM for response generation  
- **FastAPI/Flask** *(Planned: Backend API implementation)*  
- **Streamlit/React** *(Planned: Frontend UI for easy interaction)*  

## 🔧 Installation & Setup  
1️⃣ Clone the Repository  
git clone https://github.com/avish006/Rag-Project.git
cd Rag-Project

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Set Up API Keys
Create a .env file and add your OpenRouter API Key:
OPENROUTER_API_KEY=your_api_key_here

4️⃣ Run the Application
python main.py

🧠 How It Works
Document Ingestion: Users upload a document (PDF, TXT).
Chunking & Embeddings: The text is split into meaningful chunks, then converted into vector embeddings using BGE-large.
Vector Search (FAISS): Given a user query, the system finds the most relevant document chunks using FAISS.
LLM Response Generation: The retrieved context is fed into DeepSeek R1 via OpenRouter API to generate a context-aware answer.

🎯 Challenges & Optimizations
Tuned hyperparameters (temperature, top_p) to improve reasoning.
Optimized chunking strategies for better document retrieval.
Planned: Multi-step RAG to refine responses.

🔗 Contribution
Feel free to fork the repo, submit PRs, or report issues. Let's improve it together!

🚀 Future Improvements
✅ Implement FastAPI/Flask for a REST API
✅ Build a Streamlit/React UI for an interactive frontend
✅ Optimize embeddings & chunking for better retrieval

📌 License
This project is open-source under the Apache 2.0 License.
