# Capstone Project: Gemini + LangChain + ChromaDB Integration

![GenAI Banner](https://storage.googleapis.com/kaggle-media/learn/images/r3vLQYV.png)
![image_5d_genai](https://github.com/user-attachments/assets/c8a7d283-6cfb-4c23-bb2b-88c6bf4c7d0e)

This project demonstrates a Retrieval-Augmented Generation (RAG) system built for the **Kaggle's 5 Days of GenAI** course capstone, integrating Google's Gemini API with LangChain and ChromaDB.

## 📌 Project Overview

The system:
1. Processes text documents by chunking them
2. Generates embeddings using Gemini's embedding model
3. Stores embeddings in ChromaDB for efficient retrieval
4. Answers user queries by:
   - Finding relevant document chunks
   - Feeding them as context to Gemini-1.5 Flash
   - Generating accurate, context-aware responses
  
## 🛠️ Technologies Used

- **Google Gemini API** (`gemini-1.5-flash` for generation, `embedding-001` for embeddings)
- **LangChain** (Framework for chaining components)
- **ChromaDB** (Vector database for embeddings storage)
- **Python** (Primary implementation language)

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- Google API key with Gemini access
- Kaggle environment (if running on Kaggle)

### Installation
1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd your-project-folder

pip install -r requirements.txt
GOOGLE_API_KEY=your_api_key_here

### Running the Project
```markdown
## 🚀 Running the Project

Execute the main script:
```bash
python main.py
```


### Project Structure
```markdown
## 📂 Project Structure
├── .env # Environment variables
├── text.txt # Source text document
├── chroma_db/ # Chroma database storage
├── main.py # Main implementation
└── README.md # This file
```



## 💡 Example Usage

The script includes a sample question:
```python
user_question = "¿Qué estudió Mateo en la universidad?"
```


### Key Features
```markdown
## 📝 Key Features

- **Persistent Storage**: ChromaDB maintains embeddings between sessions
- **Configurable Chunking**: Adjustable chunk size and overlap
- **Transparent Scoring**: Shows similarity scores for retrieved chunks
- **Efficient Retrieval**: Returns only the most relevant context for LLM
```






https://github.com/user-attachments/assets/d50948a8-db0e-4504-94ee-52c278c4a4f7







