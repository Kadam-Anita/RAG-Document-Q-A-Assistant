# 📚 RAG Document Q&A Assistant

### Ask Questions. Retrieve Knowledge. Get Grounded Answers.

An AI-powered document question-answering application that uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from PDF documents and generate context-aware answers using a Large Language Model (LLM).

The application combines semantic search, document relevance evaluation, context refinement, and LLM-based answer generation in an interactive Streamlit interface.

## 🎯 Business Problem

Organizations maintain large volumes of information in documents such as technical manuals, training materials, business reports, policy documents, and user guides.

Finding specific information in these documents can be time-consuming and inefficient.

### Key Challenges

* **Time-consuming information retrieval:** Users must manually search through lengthy documents to find relevant information.
* **Limited contextual search:** Traditional keyword-based search may not identify information expressed using different words but similar meanings.
* **Information overload:** Large documents contain information that may not be relevant to a user's specific question.
* **Repeated queries:** Employees and support teams may repeatedly search for answers that already exist in documentation.
* **Unreliable AI responses:** An LLM answering without relevant source information may produce inaccurate or unsupported answers.

## 💡 Proposed Solution

The RAG Document Q&A Assistant enables users to upload PDF documents and ask questions in natural language.

Instead of relying solely on the LLM's pretrained knowledge, the application retrieves relevant content from the uploaded documents and provides it as context for answer generation.

### How the Solution Works

1. **Document ingestion:** Extract text from uploaded PDF files.
2. **Text chunking:** Divide document text into smaller, manageable chunks.
3. **Embedding generation:** Convert text chunks into numerical vectors using a pretrained embedding model.
4. **Vector storage:** Store the embeddings in a FAISS vector index.
5. **Semantic retrieval:** Retrieve the most relevant document chunks for the user's question.
6. **Relevance evaluation:** Use the LLM to evaluate whether retrieved chunks are useful for answering the question.
7. **Context refinement:** Filter retrieved content to retain information relevant to the question.
8. **Answer generation:** Generate a natural-language answer using the refined document context.

**Objective:** Make document-based information easier to find and understand while reducing reliance on manual searching.

## ✨ Key Features

* Upload one or multiple PDF documents.
* Ask questions using natural language.
* Generate semantic embeddings for document chunks.
* Retrieve relevant information using FAISS similarity search.
* Evaluate the relevance of retrieved document content.
* Refine context before answer generation.
* Generate answers using an Ollama-hosted LLM.
* Use LangGraph to organize the retrieval and generation workflow.
* Provide an interactive user interface through Streamlit.
* Answer using uploaded document context without requiring a web-search API.

> **Note:** RAG can reduce unsupported answers, but it does not guarantee that every response is correct.

## 🏗️ Technology Stack

| Technology              | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| Python                  | Core application development             |
| Streamlit               | Interactive web interface                |
| LangChain               | Document processing and LLM integration  |
| LangGraph               | Workflow orchestration                   |
| Hugging Face Embeddings | Convert text into vector representations |
| FAISS                   | Semantic vector similarity search        |
| Ollama                  | Run the language model locally           |
| PyPDF                   | Extract text from PDF documents          |

### Architecture

```text
             User
               |
               v
        Streamlit Interface
               |
               v
         Upload PDF Files
               |
               v
      Extract and Split Text
               |
               v
       Generate Embeddings
               |
               v
          FAISS Index
               |
               v
       Retrieve Relevant Chunks
               |
               v
      LangGraph RAG Workflow
               |
               v
       Evaluate Relevance
               |
               v
        Refine Context
               |
               v
          Ollama LLM
               |
               v
        Generated Answer
```

## ⚙️ Prerequisites

Install the following before running the application:

* Python 3.11 or another Python version compatible with all project dependencies.
* Git (optional, for cloning the repository).
* Ollama.
* A compatible Ollama language model.

**Recommended:** Use a Python virtual environment to isolate project dependencies.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kadam-Anita/RAG-Document-Q-A-Assistant.git
```

Navigate into the project folder:

```bash
cd RAG-Document-Q-A-Assistant
```

Alternatively, if you already have the project on your computer, open its folder in VS Code.

### Step 2: Create a Virtual Environment

On Windows:

```bash
py -3.11 -m venv .venv
```

Activate it using Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Or, if you are using PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed, install a compatible Python version first.

### Step 3: Install Dependencies

Ensure that `requirements.txt` exists in the project root.

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses packages such as Streamlit, LangChain, LangGraph, FAISS, Sentence Transformers, and PyPDF.

### Step 4: Install and Start Ollama

Install Ollama from its official website:

https://ollama.com/

Download the language model used by the application.

The default model configured in the application is `gemma3:1b`.

Run:

```bash
ollama pull gemma3:1b
```

Make sure Ollama is running before starting the application.

To verify the model is available, run:

```bash
ollama list
```

If you configured a different model in the application, download that model instead.

### Step 5: Run the Application

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open that address in your browser.

## 🖥️ How to Use the Application

1. Open the application in your browser.
2. Upload one or more PDF documents using the sidebar.
3. Click **Build / Refresh document index**.
4. Wait for the documents to be processed and the FAISS index to be created.
5. Enter a question related to the uploaded documents.
6. Submit the question.
7. Review the answer generated from the retrieved document context.

### Example

**Uploaded document:** A technical guide explaining machine learning concepts.

**Question:**

"What is the difference between batch normalization and layer normalization?"

**Expected behavior:**

The application searches the uploaded document for relevant content and uses that content to generate an answer.

The answer depends on the information available in the uploaded document. If the document does not contain enough information, the application is designed to indicate that limitation rather than intentionally invent facts.

## 📊 Business Use Cases

### 1. Knowledge Management

Help employees retrieve information from internal documentation.

### 2. Technical Documentation

Find explanations and instructions in technical manuals and product guides.

### 3. Customer Support

Help support teams search product documentation and troubleshooting guides.

### 4. Learning and Training

Answer questions from training materials and educational documents.

These are potential applications of the solution, not claims of existing enterprise deployment.

## 🔒 Limitations and Considerations

* The current application focuses on PDF documents.
* Scanned PDFs may require OCR before their text can be retrieved.
* Answer quality depends on document quality and retrieval relevance.
* Local response speed depends on the hardware and selected language model.
* The vector index is created from uploaded documents and may need to be rebuilt when the document collection changes.
* The local Ollama model must be available wherever the application runs.
* Authentication, document-level access control, production monitoring, and enterprise deployment require additional implementation.

**Privacy:** Do not upload confidential documents to a public repository. Keep API keys, passwords, and private configuration files out of Git.

## 🔮 Future Enhancements

* Support for Word documents, text files, and additional data sources.
* Source citations and document-page references in answers.
* Hybrid search combining keyword and semantic retrieval.
* Reranking to improve retrieval quality.
* Evaluation datasets to measure retrieval and answer quality.
* User authentication and document access control.
* Persistent vector indexes and incremental document updates.
* MCP-based tool integration and agentic workflows where appropriate.
* Deployment to a cloud environment with monitoring and security controls.

## 👩‍💻 Author

**Anita Kadam**

GitHub: https://github.com/Kadam-Anita

---

### Project Summary

The RAG Document Q&A Assistant demonstrates how retrieval-augmented generation can connect document search with natural-language answer generation.

By combining embeddings, FAISS, LangChain, LangGraph, Ollama, and Streamlit, the project provides a foundation for building document-grounded AI assistants.

**Detect relevant information. Retrieve knowledge. Generate grounded answers.**

