import os
import streamlit as st
from io import StringIO
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import tempfile
import shutil

st.set_page_config(page_title="Gemini RAG QA", layout="wide")
st.title("📚 Gemini RAG QA System")

# Step 1: Get API key from user
api_key = st.text_input("🔑 Enter your Google Gemini API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Step 2: Upload file
uploaded_file = st.file_uploader("📄 Upload a text (.txt) or PDF (.pdf) file", type=["txt", "pdf"])

if uploaded_file and api_key:
    # Read file content
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    else:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()

    # Split and deduplicate text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = list(set(text_splitter.split_text(text)))  # Remove duplicates

    # Create a temporary directory (will be deleted when Streamlit exits)
    temp_dir = tempfile.mkdtemp()

    try:
        # Embeddings and vector store
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        chroma_db = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            persist_directory=temp_dir
        )

        # Step 3: Input question
        user_question = st.text_input("❓ Ask your question:")

        if user_question:
            relevant_docs = chroma_db.similarity_search_with_score(user_question, k=3)
            all_docs = chroma_db.get(include=["documents"])
            total_embeddings = len(all_docs["documents"])

            print(f"Total number of embeddings in the Chroma database: {total_embeddings}")

            if relevant_docs:
                context_text = "\n".join([doc.page_content for doc, _ in relevant_docs])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    temperature=0,
                    max_tokens=None,
                )

                messages = [
                    ("system", "You are a helpful assistant that answers questions using the given context."),
                    ("human", f"Context: {context_text}\n\nQuestion: {user_question}"),
                ]

                with st.spinner("🤖 Generating answer..."):
                    ai_msg = llm.invoke(messages)

                st.markdown("### 💬 Answer")
                st.success(ai_msg.content)

                st.markdown("### 📌 Top 3 Relevant Chunks with Similarity Scores")
                for i, (doc, score) in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i+1}** (Similarity Score: `{score:.4f}`)")
                    st.code(doc.page_content.strip())

    finally:
        # Clean up the temporary directory after app stops
        shutil.rmtree(temp_dir, ignore_errors=True)

else:
    st.info("Please enter your API key and upload a file to continue.")
