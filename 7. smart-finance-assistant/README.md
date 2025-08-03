# 💰 Smart Finance Assistant - RAG Tutorial

A complete tutorial for building a Retrieval-Augmented Generation (RAG) system that analyzes bank statements and answers financial questions using AI.

## 🎥 Demo

https://github.com/user-attachments/assets/cd41350e-1aac-443d-ab15-d16f8ffd369a

## 🎯 What You'll Learn

This tutorial teaches you how to build a production-ready RAG application with:

- 📄 PDF document processing and text extraction
- 🔍 Vector embeddings and semantic search
- 💾 ChromaDB for vector storage
- 🤖 LLM integration with OpenRouter
- 💬 Conversation memory
- 🎨 Modern chat interface with Streamlit

## 📋 Prerequisites

- **Python 3.10** or higher
- **VS Code** (recommended) or any code editor
- **API Keys** (free tiers available):
  - [OpenRouter API key](https://openrouter.ai) - Access to multiple LLMs
  - [OpenAI API key](https://platform.openai.com) - For embeddings

## 🚀 Quick Start

Use any of your favorite editor to perform the steps.
In my case I have used Visual Studio Code with GitBash as Terminal to run bash commands.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/finance-assistant-rag.git
cd finance-assistant-rag
```

### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run smart-finance-assistant.py
```

### 5. Open in Browser

Navigate to `http://localhost:8501`

## 📖 How to Use

1. **Enter API Keys**: Add your OpenRouter and OpenAI API keys in the configuration panel
2. **Upload Documents**: Upload your bank statement PDFs
3. **Process Files**: Click "Process Documents" to build the vector database
4. **Ask Questions**: Start chatting! Try questions like:
   - "How much did I spend on groceries last month?"
   - "Show me all my subscription charges"
   - "What was my largest expense this quarter?"

## 🏗️ RAG Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PDF Documents  │────▶│  Text Splitter  │────▶│   Embeddings    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   LLM Response  │◀────│    Retriever    │◀────│  Vector Store   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│  Chat Interface │                              │   ChromaDB      │
└─────────────────┘                              └─────────────────┘
```

## ✨ Features

### Document Processing

- ✅ PDF text extraction with metadata
- ✅ Intelligent text chunking (500 chars with 50 char overlap)
- ✅ Page-level tracking and source attribution

### Retrieval System

- ✅ Semantic search using OpenAI embeddings
- ✅ Threshold-based retrieval for comprehensive results
- ✅ Query rephrasing for better accuracy

### AI Capabilities

- ✅ Multiple LLM support via OpenRouter
- ✅ Conversation memory for context-aware responses
- ✅ Financial domain-specific prompting

### User Interface

- ✅ Modern chat interface
- ✅ Real-time streaming responses
- ✅ Document upload with progress tracking
- ✅ Model selection and configuration

## 🛠️ Configuration Options

### Available Models

- **Kimi K2** (Free) - Good for testing
- **GPT-4.1 Mini** - Fast and cost-effective
- **Claude Sonnet 4** - Balanced performance
- **Gemini 2.0 Flash** - Google's latest
- **DeepSeek Chat V3** - Alternative option

### Customization

- Adjust chunk size in `process_financial_documents()`
- Modify retrieval threshold in `create_enhanced_retriever()`
- Customize the prompt template in `analyze_financial_query()`

## 🐛 Troubleshooting

### Common Issues

**"No module named 'langchain'"**

- Make sure you've activated your virtual environment
- Run `pip install -r requirements.txt` again

**"API key not found"**

- Ensure you've entered both API keys in the configuration panel
- Check that your keys are valid and have credits

**"No documents found"**

- Make sure you've uploaded and processed PDFs first
- Check that your PDFs contain readable text (not scanned images)

**ChromaDB errors**

- Delete the `./finance_db` folder and restart
- Ensure you have write permissions in the directory

## 📚 Learning Resources

### Understanding RAG

- [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Guide](https://docs.trychroma.com/)

### Next Steps

1. Try different embedding models
2. Experiment with chunk sizes
3. Add support for more document types
4. Implement advanced retrieval strategies
5. Deploy to cloud platforms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- LLM access via [OpenRouter](https://openrouter.ai/)

---

**Made with ❤️ for the RAG community**

_If this tutorial helped you, please ⭐ the repository!_
