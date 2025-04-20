import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Access environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"API_KEY: {GOOGLE_API_KEY}")  # Optional

# File paths
text_file = 'text.txt'
chroma_dir = "chroma_db"

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if Chroma DB already exists
if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
    print("🔁 Loading existing Chroma DB...")
    chroma_db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)
else:
    print("📦 Creating new Chroma DB...")

    # Read full text
    with open(text_file, 'r', encoding='UTF-8') as file:
        text = file.read()

    # Split text into 200-character chunks
    text_splitter = CharacterTextSplitter(
        separator="",            # no separator, split by character count
        chunk_size=200,          # split every 200 characters
        chunk_overlap=10          # no overlap
    )
    chunks = text_splitter.split_text(text)
    print(f"📄 Total chunks: {len(chunks)}")

    # Create and save the embeddings
    chroma_db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=chroma_dir
    )
    chroma_db.persist()
    print("✅ Embeddings stored in Chroma DB.")

# Question to answer
user_question = "¿Qué estudió Mateo en la universidad?"

# Find the most relevant chunks
# Find the most relevant chunks
relevant_docs = chroma_db.similarity_search_with_score(user_question, k=3)
relevant_text = "\n".join([doc.page_content for doc, _ in relevant_docs])  # Extract only the document content

# Print the results
print("-" * 40)
for i, (doc, score) in enumerate(relevant_docs):
    print(f"🔍 Chunk {i+1}:")
    print(doc.page_content)  # Display the content of each relevant document chunk
    print(f"Score: {score:.4f}")  # Print the similarity score, rounded to 4 decimal places
print("-" * 40)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
)

# Construct prompt with retrieved chunks as context
messages = [
    ("system", "You are a helpful assistant that answers questions using the given context."),
    ("human", f"Context: {relevant_text}\n\nQuestion: {user_question}"),
]

# Get and print the response
ai_msg = llm.invoke(messages)
print("\n💬 Answer:")
print(ai_msg.content)
