# 🤖 RAG from Scratch: Complete Tutorial Course

A comprehensive educational resource for learning Retrieval-Augmented Generation (RAG) systems from the ground up. This project combines video course transcripts with hands-on Python implementations to teach RAG fundamentals and advanced techniques.

## 📚 What You'll Learn

This course takes you through the complete RAG pipeline, from basic concepts to production-ready implementations:

- **Document Loading & Processing** - Handle PDFs, markdown, and various file formats
- **Text Splitting Strategies** - Optimize chunking for better retrieval performance  
- **Vector Stores & Embeddings** - Build efficient document indexing systems
- **Retrieval Techniques** - Master similarity search, MMR, metadata filtering, and self-query
- **Question Answering** - Create complete RAG applications with LangChain

## 🎯 Perfect For

- **AI Engineers** building RAG applications
- **Data Scientists** exploring document retrieval systems
- **Developers** integrating LLMs with external knowledge bases
- **Students** learning modern AI/ML techniques
- **Anyone** curious about how ChatGPT-style systems work with custom data

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain_community openai chromadb
```

### Basic RAG Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# 1. Split your documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 2. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 3. Retrieve relevant documents
retriever = vectorstore.as_retriever()
relevant_docs = retriever.get_relevant_documents("your question")
```

## 📖 Course Modules

### Module 1: RAG Fundamentals
Learn why RAG exists and how it solves the limitations of pre-trained LLMs with private or recent data.

### Module 2: Document Loading
Master loading various document formats (PDFs, markdown, web pages) using LangChain's document loaders.

### Module 3: Text Splitting Strategies
Understand chunking techniques that make or break RAG performance:
- Character-based splitting
- Token-aware splitting  
- Structure-preserving splits
- Optimal chunk sizes and overlap

### Module 4: Vector Stores & Embeddings
Build efficient document indexes using:
- OpenAI embeddings
- Chroma vector database
- Persistence and optimization

### Module 5: Advanced Retrieval
Go beyond basic similarity search:
- **MMR (Maximal Marginal Relevance)** - Diverse results
- **Metadata Filtering** - Precise source targeting
- **Self-Query Retrieval** - Natural language filters
- **Contextual Compression** - Focused relevant content

### Module 6: Question Answering
Build complete RAG applications with prompt engineering and chain optimization.

## 🛠️ Key Features

- **Step-by-step tutorials** with real code examples
- **Visual diagrams** explaining RAG architecture
- **Practical tips** for production deployments
- **Common pitfalls** and how to avoid them
- **Performance optimization** techniques
- **LinkedIn content generator** for sharing RAG insights

## 📈 What Makes This Different

Unlike basic RAG tutorials, this course covers:

✅ **Real-world challenges** like duplicate content and poor chunking  
✅ **Multiple retrieval strategies** beyond simple similarity search  
✅ **Production considerations** for scaling RAG systems  
✅ **Hands-on examples** with actual PDF documents  
✅ **Advanced techniques** like contextual compression and self-query  

## 🎓 Learning Path

1. **Start here**: Read the Introduction transcript to understand RAG motivation
2. **Get hands-on**: Open `rag-document-splitting.ipynb` for interactive learning
3. **Build skills**: Work through each module transcript + notebook
4. **Practice**: Try the examples with your own documents
5. **Share knowledge**: Use the LinkedIn generator to create content from your learnings

## 💡 Pro Tips for RAG Success

**Chunking Strategy**
- Start with 1000 characters, 100 overlap
- Test with your specific documents
- Smaller chunks for Q&A, larger for summarization

**Retrieval Optimization**  
- Use MMR to reduce duplicate content
- Add metadata filtering for precision
- Compress results to focus on relevant parts

**Performance Monitoring**
- Track retrieval accuracy with real user queries
- A/B test different chunk sizes
- Monitor for edge cases and failure modes

## 🤝 Contributing

Found a bug or have suggestions? We'd love to hear from you:

- 🐛 **Bug reports**: Open an issue with detailed steps to reproduce
- 💡 **Feature requests**: Suggest improvements or new examples  
- 📝 **Documentation**: Help improve explanations or add examples
- 🔄 **Code improvements**: Submit PRs for better implementations

## 📬 Connect & Share

Built something cool with RAG? We'd love to see it! 

- Share your RAG projects using `#RAGfromScratch`
- Connect with the community of RAG builders
- Use the LinkedIn content generator to share your insights

## ⚖️ License

This educational content is open source. Feel free to use, modify, and share for learning purposes.

---