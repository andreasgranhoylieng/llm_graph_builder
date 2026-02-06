# 🧠 Neo4j LLM Knowledge Graph Builder

A robust, enterprise-grade tool for building high-quality knowledge graphs from unstructured documents (PDF/TXT) using Large Language Models (LLMs). This project focuses on large-scale batch processing, reliability, and follows a clean architectural pattern.

## 🌟 Features

- **Large-Scale Batch Processing**: Efficiently handles hundreds of documents with checkpointing.
- **Progress Tracking**: Real-time progress bars, ETA, and processing statistics.
- **Reliability**: Automatic retries for API failures and rate-limiting.
- **Interactive Chat**: Query your generated knowledge graph directly from the CLI.
- **Modern Architecture**: Implements the "Concentric Circles" Domain-Driven Design pattern.

## 🏗️ Architecture

This project strictly adheres to the **Concentric Circles** architecture:

1.  **Outer Circle (Router/CLI)**: `main.py` serves as the entry point, handling user interaction and routing to the controller.
2.  **Middle Circle (Controller)**: `src/controllers/` orchestrates the flow of data between the CLI and the inner logic, validating inputs and mapping responses.
3.  **Inner Circle (Application & Infrastructure)**:
    - **Application Layer**: `src/services/` contains the core business logic for extraction and graph construction.
    - **Infrastructure Layer**: `src/repositories/` handles data persistence (Neo4j) and LLM integrations.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Neo4j Database (AuraDB or Local)
- OpenAI API Key

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/llm_graph_builder.git
    cd llm_graph_builder
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    ```

### Usage

Run the main CLI tool:
```bash
python main.py
```

Choose from the following options:
- **Option 1**: Batch process documents in a folder (Recommended).
- **Option 3**: View real-time status of the current processing job.
- **Option 7**: Start an interactive chat session with your knowledge graph.

## 🛠️ Tech Stack

- **Orchestration**: LangChain
- **LLM**: OpenAI GPT models
- **Graph Database**: Neo4j
- **Utilities**: pypdf, python-dotenv, tqdm, tenacity

## ⚖️ License

MIT License - See [LICENSE](LICENSE) for details.
