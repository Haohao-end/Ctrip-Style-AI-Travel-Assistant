# MCP Agent Orchestrator

The **MCP Agent Orchestrator** is a professional-grade Python implementation of the **Model Context Protocol (MCP)**. It provides a structured environment for Large Language Models (LLMs) to interact with external tools and knowledge bases through a standardized communication layer. The project utilizes `FastMCP` for server-side tool definitions and an asynchronous client-side bridge to OpenAI-compatible interfaces.

## System Architecture

The project follows a decoupled client-server architecture:

1. **MCP Client**: Acts as the orchestrator. It manages the lifecycle of the MCP server, performs tool discovery, handles LLM completions, and executes tool calls returned by the model.
2. **MCP Servers**: Independent services (Weather, RAG) that expose specific functions to the client via the Model Context Protocol.
3. **Transport Layer**: Uses Standard Input/Output (StdIO) for high-performance, local inter-process communication.

## Core Components

### 1. Intelligent Client Bridge

The client implementation (`rag_agent.py`, `client.py`) facilitates:

* Asynchronous lifecycle management using `AsyncExitStack`.
* Automatic tool schema conversion for OpenAI-compatible function calling.
* Persistent conversation state and multi-turn reasoning loops.

### 2. Weather Service Server

The weather server (`server.py`) demonstrates real-time API integration:

* Integration with external REST APIs (WeatherAPI).
* Data normalization and formatting for LLM consumption.
* Asynchronous request handling using `httpx`.

### 3. RAG Knowledge Server

The RAG server (`rag_server.py`) provides advanced document intelligence:

* **Data Ingestion**: Support for PDF and TXT formats using `LangChain`.
* **Vector Database**: Persistent storage via `ChromaDB`.
* **Search Optimization**: Implements Maximal Marginal Relevance (MMR) for diverse information retrieval.
* **Embeddings**: Integration with HuggingFace transformer models.

## Project Structure

```text
├── client.py             # Standard MCP client implementation
├── rag_agent.py          # Specialized agent for RAG operations
├── server.py             # Weather service MCP server
├── rag_server.py         # RAG knowledge base MCP server
├── test.py               # Connectivity test for LLM API
├── .env                  # Environment configuration
└── data/
    ├── rag_db/           # Vector store persistence directory
    └── doupocangqiong.txt # Sample knowledge base source

```

## Technical Stack

* **Protocol**: Model Context Protocol (MCP)
* **LLM Interface**: OpenAI SDK (Compatible with Qwen/DashScope)
* **RAG Framework**: LangChain
* **Vector Store**: ChromaDB
* **Communication**: Asynchronous I/O (asyncio)

## Installation

### Prerequisites

* Python 3.10+
* Virtual environment (recommended)

### Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
API_KEY=your_llm_api_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen-plus
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

```

### Dependencies

```bash
pip install mcp langchain langchain-community langchain-openai chromadb httpx python-dotenv openai

```

## Usage

### Running the Weather Agent

To start the client and connect it to the weather server:

```bash
python client.py server.py

```

### Running the RAG Agent

To initialize the knowledge base and start the RAG-enabled agent:

```bash
python rag_agent.py --server_script rag_server.py

```

## Protocol Implementation Details

The implementation strictly adheres to the MCP specification:

1. **Initialization**: Client initializes the session and retrieves tool manifests.
2. **Tool Discovery**: LLM is informed of available functions via JSON schema.
3. **Execution**: Client intercepts `tool_calls`, executes the corresponding server function, and returns the result to the LLM for final synthesis.
