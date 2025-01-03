# Contextual Document Comparison and Search Application

This application enables users to upload PDF files, extract their content, store and retrieve data from a vector database, and perform contextual comparisons or searches. It utilizes various machine learning and natural language processing tools for efficient document chunking, summarization, and comparison.

## Project Structure

### 1. **Compare_Documents.py**
- **Purpose**: Facilitates contextual comparisons between two uploaded PDF files based on user queries.
- **Features**:
  - Upload and process two PDF files.
  - Chunk and extract texts, tables, and images from PDFs.
  - Store extracted data into MongoDB Atlas Vector Database.
  - Use Jina embeddings for vector search.
  - Generate a detailed comparison using LangChain's prompting and a generative AI model.
- **Key Libraries**: Streamlit, LangChain, MongoDB, Jina Embeddings.

### 2. **Search.py**
- **Purpose**: Allows searching across two uploaded documents for relevant information based on user queries.
- **Features**:
  - Upload and process two PDF files.
  - Chunk documents and extract texts, tables, and images.
  - Summarize extracted data.
  - Store summaries and raw data into MongoDB Atlas Vector Database.
  - Use vector search to find top relevant segments matching the query.
  - Display results in Streamlit tabs.
- **Key Libraries**: Streamlit, LangChain, Jina Embeddings, MongoDB.

### 3. **chunking.py**
- **Purpose**: Provides utilities for chunking, summarizing, and storing document content in a vector database.
- **Key Classes**:
  - **Chunking**: Extracts elements like text, tables, and images from PDFs.
  - **Summarize**: Summarizes texts, tables, and images.
  - **StoreInVectorDB**: Manages storing data and summaries into MongoDB Atlas Vector Database using Jina embeddings.

### 4. **Home.py**
- **Purpose**: Serves as the homepage for the application, allowing users to upload a single PDF file and ask questions based on its content.
- **Features**:
  - Upload and process a PDF file.
  - Extract, summarize, and store document data.
  - Perform a vector search for answering user queries.
- **Key Libraries**: Streamlit, LangChain, MongoDB, Jina Embeddings.

---

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install dependencies using `pip install -r requirements.txt`.
3. Start the Streamlit application:
   - For comparison: `streamlit run Compare_Documents.py`
   - For search: `streamlit run Search.py`
   - For homepage: `streamlit run Home.py`
4. Upload PDF files and interact with the application.

## Requirements
- Python 3.9+
- Streamlit
- LangChain
- MongoDB Atlas
- Jina AI
- Unstructured (for PDF processing)

## Features in Action
- **Chunking and Summarization**:
  - Extract content like text, tables, and images.
  - Generate concise summaries for enhanced search and comparison.
- **Vector Search**:
  - Retrieve top matching segments using embeddings.
- **Generative AI Integration**:
  - Perform advanced contextual analysis with Google Gemini.
