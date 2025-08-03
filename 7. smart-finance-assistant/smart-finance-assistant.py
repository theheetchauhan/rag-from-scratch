"""
===========================================
SMART FINANCE ASSISTANT - RAG TUTORIAL
===========================================

A complete tutorial for building a Retrieval-Augmented Generation (RAG) system 
that analyzes bank statements and answers financial questions using AI.

This script demonstrates the full RAG pipeline with detailed explanations:
1. Document Loading → Load PDF bank statements
2. Text Splitting → Break documents into chunks
3. Embedding → Convert text to vectors
4. Storage → Save vectors in ChromaDB
5. Retrieval → Find relevant chunks for queries
6. Generation → Create answers using LLM + context
7. Memory → Maintain conversation history

📚 For setup instructions, prerequisites, and video demo, see README.md

🎓 LEARNING PATH:
This file contains extensive inline comments explaining:
- Why each component is needed
- How the RAG pipeline works
- Best practices and alternatives
- Common pitfalls to avoid

Start by reading through the comments, then run the app to see it in action!

"""

import os
import streamlit as st
import requests
import chromadb
from chromadb.config import Settings

# ==============================================================================
# IMPORTS SECTION - Core RAG Components
# ==============================================================================
# These imports bring in the essential building blocks for our RAG system:
# - Chroma: Vector database for storing embeddings
# - Document loaders: For reading PDFs
# - Text splitters: For chunking documents
# - LangChain components: For building the RAG pipeline
# - Memory: For conversation history
# ==============================================================================

from langchain_chroma import Chroma  # Vector store for semantic search
from langchain_community.document_loaders import PyPDFLoader  # PDF document loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking
from langchain_core.prompts import ChatPromptTemplate  # Structured prompt creation
from langchain_core.output_parsers import StrOutputParser  # Parse LLM outputs
from langchain_core.runnables import RunnablePassthrough  # Chain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI integrations
from langchain.memory import ConversationBufferMemory  # Conversation memory
from langchain.chains import ConversationChain  # Conversation management
import time  # For streaming effects


# ==============================================================================
# CUSTOM LLM CLASS - OpenRouter Integration
# ==============================================================================
# This class creates a custom LLM interface for OpenRouter, which provides
# access to multiple AI models through a single API. OpenRouter acts as a
# gateway to models like Claude, GPT-4, and others.
#
# WHY CUSTOM CLASS?
# - LangChain doesn't have built-in OpenRouter support
# - Allows flexibility in model selection
# - Handles API communication and error management
# ==============================================================================

class OpenRouterLLM:
    """
    Custom LLM class for OpenRouter API integration.
    
    This class wraps the OpenRouter API to make it compatible with LangChain.
    OpenRouter provides access to multiple LLM providers through one API.
    
    Attributes:
        api_key: Your OpenRouter API key
        model: The model to use (e.g., 'anthropic/claude-3-haiku')
        base_url: OpenRouter API endpoint
    """
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        """
        Initialize the OpenRouter LLM client.
        
        Args:
            api_key: OpenRouter API key for authentication
            model: Model identifier (see openrouter.ai for options)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"  # OpenRouter endpoint
    
    def invoke(self, input_data) -> str:
        """
        Generate response using OpenRouter API.
        
        This method handles various input formats from LangChain and sends
        them to OpenRouter for processing. It includes comprehensive error
        handling for robust operation.
        
        Args:
            input_data: Can be string, ChatPromptValue, or message object
            
        Returns:
            str: Generated response from the LLM
        """
        try:
            # ===== INPUT HANDLING =====
            # LangChain can pass different types of inputs, so we need to
            # handle each case to extract the actual text content
            
            if hasattr(input_data, 'to_string'):
                # ChatPromptValue object - convert to string
                messages_content = input_data.to_string()
            elif hasattr(input_data, 'messages'):
                # Message list - extract content from first message
                messages_content = input_data.messages[0].content if input_data.messages else str(input_data)
            elif hasattr(input_data, 'content'):
                # Direct message object - get content attribute
                messages_content = input_data.content
            else:
                # Simple string - use as-is
                messages_content = str(input_data)
            
            # ===== API REQUEST SETUP =====
            # Prepare headers for authentication and content type
            headers = {
                "Authorization": f"Bearer {self.api_key}",  # API key authentication
                "Content-Type": "application/json"
            }
            
            # ===== REQUEST PAYLOAD =====
            # Structure the request according to OpenRouter's API format
            data = {
                "model": self.model,  # Which AI model to use
                "messages": [{"role": "user", "content": messages_content}],  # User message
                "temperature": 0.7,  # Creativity level (0=deterministic, 1=creative)
                "max_tokens": 1000  # Maximum response length
            }
            
            # ===== API CALL =====
            # Send request to OpenRouter and wait for response
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            result = response.json()
            
            # Extract the generated text from API response
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            # Network or API errors
            error_msg = f"API request failed: {str(e)}"
            st.error(error_msg)
            return "Sorry, I encountered a network error. Please check your API key and try again."
        except KeyError as e:
            # Unexpected response format
            error_msg = f"Unexpected API response format: {str(e)}"
            st.error(error_msg)
            return "Sorry, I received an unexpected response from the API."
        except Exception as e:
            # Catch-all for other errors
            error_msg = f"Error generating response: {str(e)} (Input type: {type(input_data)})"
            st.error(error_msg)
            return "Sorry, I encountered an error processing your request."


# ==============================================================================
# COMPONENT INITIALIZATION - Setting up RAG Infrastructure
# ==============================================================================
# This function initializes the two core components needed for RAG:
# 1. Embedding Model: Converts text to vectors (numbers)
# 2. Vector Database: Stores and searches these vectors
#
# Think of embeddings as "meaning coordinates" - similar meanings have
# similar coordinates, allowing us to find relevant content.
# ==============================================================================

def initialize_components():
    """
    Initialize embedding model and vector database for RAG.
    
    This is the foundation of our RAG system. It sets up:
    - OpenAI embeddings for converting text to vectors
    - ChromaDB for storing and searching those vectors
    
    Returns:
        tuple: (embedding_model, database) or (None, None) if error
    """
    # ===== API KEY VALIDATION =====
    # Both APIs are required - OpenRouter for generation, OpenAI for embeddings
    if not st.session_state.get("openrouter_api_key"):
        st.error("Please enter your OpenRouter API key in the sidebar.")
        return None, None
    
    if not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None, None
    
    try:
        # ===== EMBEDDING MODEL SETUP =====
        # OpenAI's text-embedding-3-small is cost-effective and fast
        # It converts text into 1536-dimensional vectors
        embedding_model = OpenAIEmbeddings(
            api_key=st.session_state.openai_api_key,
            model="text-embedding-3-small"  # Smaller, faster embedding model
        )
        
        # ===== DATABASE DIRECTORY =====
        # Create a local directory to persist our vector database
        db_path = "./finance_db"
        os.makedirs(db_path, exist_ok=True)  # Create if doesn't exist
        
        # ===== CHROMADB CONFIGURATION =====
        # ChromaDB is an open-source vector database perfect for RAG
        chroma_settings = Settings(
            persist_directory=db_path,  # Where to save the database
            anonymized_telemetry=False  # Disable usage tracking
        )
        
        # ===== CREATE DATABASE CLIENT =====
        # PersistentClient ensures data is saved between sessions
        client = chromadb.PersistentClient(
            path=db_path,
            settings=chroma_settings
        )
        
        # ===== INITIALIZE VECTOR STORE =====
        # This creates a "collection" (like a table) for our financial documents
        db = Chroma(
            collection_name="finance_statements",  # Name for our document collection
            embedding_function=embedding_model,  # How to convert text to vectors
            client=client,
            persist_directory=db_path  # Where to save everything
        )
        
        return embedding_model, db
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None


# ==============================================================================
# DOCUMENT FORMATTING - Preparing Retrieved Content
# ==============================================================================
# After retrieval, we need to format the documents into a single string
# that can be used as context for the LLM. This simple function handles that.
# ==============================================================================

def format_docs(docs):
    """
    Formats a list of document objects into a single string.
    
    When we retrieve documents from the vector store, they come as objects.
    This function extracts just the text content and combines them.
    
    Args:
        docs: List of Document objects from vector store
        
    Returns:
        str: Combined text content separated by double newlines
    """
    # Join all document contents with double newline for readability
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================================================================
# DOCUMENT PROCESSING PIPELINE - The Heart of RAG Data Preparation
# ==============================================================================
# This is where we implement the full document processing pipeline:
# 1. Load PDFs
# 2. Extract text with metadata
# 3. Split into optimal chunks
# 4. Store in vector database
#
# Good chunking is CRITICAL for RAG performance. Too small = lost context.
# Too large = irrelevant content mixed with relevant.
# ==============================================================================

def process_financial_documents(uploaded_files, db):
    """
    Processes and adds uploaded financial PDF files to the vector database.
    
    This function implements the document ingestion pipeline:
    1. Save uploaded files temporarily
    2. Load PDFs and extract text
    3. Split text into chunks
    4. Generate embeddings
    5. Store in vector database
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
        db: ChromaDB instance for storing processed documents
    """
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    # ===== PROGRESS TRACKING =====
    # Visual feedback for users during processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each uploaded file
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # ===== TEMPORARY FILE HANDLING =====
        # Streamlit uploads exist in memory, but PyPDFLoader needs a file path
        # So we temporarily save the file to disk
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())  # Write uploaded content

        try:
            # ===== PDF LOADING =====
            # PyPDFLoader extracts text while preserving page structure
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()  # Returns list of Document objects

            # ===== METADATA ENRICHMENT =====
            # Adding metadata helps with retrieval and source tracking
            doc_metadata = []
            doc_content = []
            
            for i, doc in enumerate(data):
                # Enhance each page's metadata
                metadata = doc.metadata.copy()
                metadata.update({
                    'source_file': uploaded_file.name,  # Track which file this came from
                    'page_number': i + 1,  # Human-readable page numbers
                    'document_type': 'bank_statement'  # Document classification
                })
                doc_metadata.append(metadata)
                doc_content.append(doc.page_content)

            # ===== TEXT SPLITTING STRATEGY =====
            # RecursiveCharacterTextSplitter is the Swiss Army knife of splitters
            # It tries to split on paragraphs first, then sentences, then words
            # This maintains semantic coherence better than simple character splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Characters per chunk - good for transaction data
                chunk_overlap=50,  # Overlap prevents cutting mid-transaction
                length_function=len,  # How to measure chunk size
                separators=["\n\n", "\n", " ", ""]  # Split priority order
            )
            
            # Create document chunks with associated metadata
            chunks = text_splitter.create_documents(doc_content, doc_metadata)

            # ===== VECTOR STORAGE =====
            # This is where the magic happens:
            # 1. Each chunk is converted to an embedding vector
            # 2. Vectors are stored in ChromaDB
            # 3. ChromaDB indexes them for fast similarity search
            db.add_documents(chunks)
            
            # Update progress bar
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # ===== CLEANUP =====
            # Always remove temporary files to avoid disk clutter
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    status_text.text("Processing complete!")
    progress_bar.progress(1.0)


# ==============================================================================
# ENHANCED RETRIEVER - Smart Document Search
# ==============================================================================
# The retriever is responsible for finding relevant documents for a query.
# We use similarity search with a threshold to get ALL relevant content,
# not just the top K results. This ensures comprehensive answers.
# ==============================================================================

def create_enhanced_retriever(db, embedding_model):
    """
    Create retriever that can fetch all relevant documents.
    
    Traditional retrievers return only top-K results. Our enhanced version
    uses similarity thresholds to return ALL documents above a relevance
    score, ensuring we don't miss important information.
    
    Args:
        db: Vector database instance
        embedding_model: Model for converting queries to vectors
        
    Returns:
        Retriever object or None if error
    """
    if not db:
        return None
    
    # ===== RETRIEVAL STRATEGY =====
    # similarity_score_threshold: Returns all docs above threshold
    # This is better than top-K for comprehensive financial analysis
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",  # Threshold-based retrieval
        search_kwargs={
            'score_threshold': 0.1,  # Very low = get all remotely relevant docs
            'k': 100  # High backup limit to ensure we get everything
        }
    )
    
    return retriever


# ==============================================================================
# MAIN RAG QUERY PROCESSING - Where Everything Comes Together
# ==============================================================================
# This is the complete RAG pipeline in action:
# 1. Rephrase query for better retrieval
# 2. Retrieve relevant documents
# 3. Format context
# 4. Generate response with context
# 5. Update conversation memory
#
# This function demonstrates advanced RAG techniques like query rephrasing
# and conversation memory integration.
# ==============================================================================

def analyze_financial_query(query: str, db, embedding_model):
    """
    Processes a financial query using enhanced RAG with memory.
    
    This is the main RAG pipeline that:
    1. Rephrases queries for better retrieval
    2. Retrieves relevant bank statement data
    3. Maintains conversation context
    4. Generates accurate, contextual responses
    
    Args:
        query: User's financial question
        db: Vector database with financial documents
        embedding_model: Model for embeddings
        
    Returns:
        str: AI-generated response based on retrieved context
    """
    if not db:
        return "Please upload bank statements first and ensure both API keys are set."
    
    # ===== CONVERSATION MEMORY SETUP =====
    # Memory allows the AI to remember previous questions and answers
    # This enables follow-up questions like "What about last month?"
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",  # Key for accessing history
            return_messages=True  # Return as message objects
        )
    
    memory = st.session_state.conversation_memory
    
    # Create retriever for this query
    retriever = create_enhanced_retriever(db, embedding_model)
    
    if not retriever:
        return "Error creating retriever. Please check your setup."

    try:
        # ===== QUERY REPHRASING =====
        # Users often ask vague questions. Rephrasing adds financial keywords
        # and specificity to improve retrieval accuracy.
        # Example: "spending?" → "transaction amounts merchant names dates bank statement"
        rephrasing_prompt = f"""
        Rephrase this query to better find relevant bank statement transactions. 
        Add financial keywords and be specific about transaction types, amounts, dates, or merchants.

        Original query: {query}

        Rephrased query (one sentence):
        """

        # Use LLM to rephrase the query
        llm = OpenRouterLLM(
            api_key=st.session_state.openrouter_api_key, 
            model=st.session_state.get("selected_model", "moonshotai/kimi-k2:free")
        )
        rephrased_query = llm.invoke(rephrasing_prompt)

        # ===== DOCUMENT RETRIEVAL =====
        # Use the rephrased query to find relevant chunks from our vector store
        retrieved_docs = retriever.invoke(rephrased_query)
        
        # Check if we found any relevant documents
        if len(retrieved_docs) == 0:
            return "No relevant documents found in your bank statements. Please make sure you have uploaded and processed your bank statement PDFs."
        
        # Format retrieved documents into context string
        context = format_docs(retrieved_docs)
        
        # ===== CONVERSATION HISTORY EXTRACTION =====
        # Get previous conversation to maintain context
        chat_history = memory.load_memory_variables({})

        # Convert message objects to readable format
        if chat_history.get("chat_history"):
            if isinstance(chat_history["chat_history"], list):
                history_str = ""
                for msg in chat_history["chat_history"]:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                        history_str += f"{role}: {msg.content}\n"
            else:
                history_str = str(chat_history["chat_history"])
        else:
            history_str = ""

        # ===== PROMPT ENGINEERING =====
        # This prompt template structures how we present information to the LLM
        # Good prompts are crucial for accurate, helpful responses
        PROMPT_TEMPLATE = """
        You are a professional financial assistant specializing in personal finance analysis.
        
        Previous conversation context:
        {chat_history}
        
        Based on the following bank statement data:
        {context}

        Answer this financial question: {question}

        Guidelines:
        - Provide specific amounts and dates when available
        - Identify spending patterns and trends
        - Highlight unusual transactions
        - Use appropriate currency symbols
        - Be precise with calculations
        - If asked about subscriptions, identify recurring charges
        - Categorize by merchant or type for spending analysis
        - Only provide factual analysis, not financial advice
        - If information is not available, clearly state that
        - Do not show your thinking process, only provide the final answer
        - When asked follow-up questions, refer to the previous conversation context
        
        Answer:
        """

        # Create prompt template object
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # Initialize LLM for response generation
        llm = OpenRouterLLM(
            api_key=st.session_state.openrouter_api_key,
            model=st.session_state.get("selected_model", "moonshotai/kimi-k2:free")
        )

        # ===== PROMPT FORMATTING =====
        # Fill in the template with actual values
        formatted_prompt = prompt_template.format(
            chat_history=history_str,  # Previous conversation
            context=context,  # Retrieved bank statement data
            question=query  # Current question
        )
        
        # ===== RESPONSE GENERATION =====
        # Send the complete prompt to the LLM
        response = llm.invoke(formatted_prompt)
        
        # ===== MEMORY UPDATE =====
        # Save this Q&A pair for future context
        memory.save_context({"input": query}, {"output": response})
        
        return response
        
    except Exception as e:
        return f"Error processing your query: {str(e)}"


# ==============================================================================
# MAIN APPLICATION - Streamlit UI with Tutorial Comments
# ==============================================================================
# This section creates the user interface using Streamlit.
# The layout mimics popular chat applications with:
# - Configuration panel on the left (sidebar)
# - Chat interface on the right (main area)
# - Modern, intuitive design with minimal clutter
# ==============================================================================

def main():
    # Configure the main page settings - this must be the first Streamlit command
    st.set_page_config(
        page_title="Finance Assistant",  # Browser tab title
        page_icon="💰",                  # Browser tab icon
        layout="wide"                    # Use full width instead of centered
    )

    # ========== SESSION STATE INITIALIZATION ==========
    # Session state persists data across reruns (when user interacts with widgets)
    # Initialize chat history if it doesn't exist - this runs only once per session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hi! I'm your finance assistant. Upload your bank statements and ask me anything!"
            }
        ]

    # ========== SIDEBAR - Left Panel Configuration ==========
    # Sidebar contains all controls and settings, keeping main area clean for chat
    with st.sidebar:
        # App title in sidebar instead of main area for space efficiency
        st.title("💰 Finance Assistant")
        
        # ========== CHAT CONTROLS ==========
        # Clear button placed prominently at top for easy access
        # When clicked, this resets the entire conversation
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            # Reset messages to initial welcome message
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "Hi! I'm your finance assistant. Upload your bank statements and ask me anything!"
                }
            ]
            # Clear conversation memory if it exists (for advanced chat features)
            if "conversation_memory" in st.session_state:
                st.session_state.conversation_memory.clear()
            # Rerun the app to refresh the interface immediately
            st.rerun()
        
        # ========== FEATURES OVERVIEW ==========
        # Show users what the app can do - placed early for immediate context
        st.subheader("ℹ️ Features")
        st.caption(
            "💳 Spending analysis  \n"      # \n creates line breaks in caption
            "🔄 Subscription tracking  \n"
            "💬 Follow-up questions  \n"
            "📊 Trend insights"
        )
        
        # ========== API CONFIGURATION ==========
        # Users need to provide API keys for the AI models to work
        st.subheader("🔧 Setup")
        
        # Password input fields hide the API keys for security
        openrouter_api_key = st.text_input("OpenRouter API", type="password")
        openai_api_key = st.text_input("OpenAI API", type="password")

        # Model selection dropdown - maps user-friendly names to API model IDs
        model_map = {
            "Kimi K2 (Free)": "moonshotai/kimi-k2:free",        # Free option
            "GPT-4.1 Mini": "openai/gpt-4.1-mini",             # Fast and efficient
            "Claude Sonnet 4": "anthropic/claude-sonnet-4",     # Current model
            "Gemini 2.0 Flash": "google/gemini-2.0-flash-001", # Google's latest
            "DeepSeek Chat V3": "deepseek/deepseek-chat-v3-0324" # Alternative option
        }
        model_name = st.selectbox("Model", list(model_map.keys()), index=0)
        
        # Save configuration to session state when both API keys are provided
        if openrouter_api_key and openai_api_key and model_name is not None:
            # Store API keys in session state for use throughout the app
            st.session_state.openrouter_api_key = openrouter_api_key
            st.session_state.openai_api_key = openai_api_key
            # Store both the internal model ID and user-friendly name
            st.session_state.selected_model = model_map[model_name]
            st.session_state.selected_model_name = model_name
            # Show confirmation that settings are saved
            st.success("✅ Keys Saved")

        # ========== DOCUMENT UPLOAD AND PROCESSING ==========
        # Users can upload PDF bank statements for analysis
        st.subheader("📤 Documents")
        
        # ========== DATABASE STATUS INDICATOR ==========
        # Show current status of the document database to guide user actions
        embedding_model, db = initialize_components()
        
        if db is not None and hasattr(db, '_collection') and db._collection.count() > 0:
            # Database exists and contains processed documents
            doc_count = db._collection.count()
            st.success(f"✅ Database Ready ({doc_count} docs)")
        else:
            # No documents processed yet - user needs to upload and process files
            st.warning("⚠️ No documents processed")
        
        # File uploader widget - accepts multiple PDF files
        uploaded_files = st.file_uploader(
            "Upload PDFs", 
            type=["pdf"],                    # Restrict to PDF files only
            accept_multiple_files=True       # Allow multiple file selection
        )
        
        # Process documents button - separate from upload for user control
        if st.button("Process Docs"):
            # Validation: ensure files are uploaded before processing
            if not uploaded_files:
                st.warning("Upload files first.")
            else:
                # Show spinner during processing (can take time for large files)
                with st.spinner("Processing..."):
                    # Extract and process financial data from uploaded PDFs
                    process_financial_documents(uploaded_files, db)
                    st.success("✅ Processed!")
                    # Refresh the app to update the interface
                    st.rerun()

    # ========== MAIN CHAT INTERFACE ==========
    # This section creates the chat area in the main content space
    
    # Initialize AI components needed for chat functionality
    embedding_model, db = initialize_components()

    # ========== DISPLAY CHAT HISTORY ==========
    # Loop through all stored messages and display them
    for msg in st.session_state.messages:
        # Create a chat message bubble with role-based styling
        with st.chat_message(msg["role"]):  # "user" or "assistant"
            st.write(msg["content"])

    # ========== CHAT INPUT AND PROCESSING ==========
    # Chat input widget at bottom of page (Streamlit automatically positions it)
    if prompt := st.chat_input("Ask a financial question..."):
        
        # ========== VALIDATION CHECKS ==========
        # Ensure user has provided necessary configuration before processing
        if not st.session_state.get("openrouter_api_key") or not st.session_state.get("openai_api_key"):
            st.info("Enter both API keys in the sidebar to continue.")
            st.stop()  # Halt execution if API keys missing
            
        # Ensure documents have been processed before answering questions
        if db is None:
            st.info("Upload and process PDFs first.")
            st.stop()  # Halt execution if no documents processed

        # ========== ADD USER MESSAGE ==========
        # Store user's question in chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user's message immediately
        with st.chat_message("user"):
            st.write(prompt)

        # ========== GENERATE AND DISPLAY AI RESPONSE ==========
        # Create assistant message bubble and generate response
        with st.chat_message("assistant"):
            # Show spinner while AI processes the query (can take several seconds)
            with st.spinner("Analyzing..."):
                # Call the main analysis function with user's question and documents
                response = analyze_financial_query(prompt, db, embedding_model)
            
            # Display the AI's response
            st.write(response)
            
            # Store assistant's response in chat history for persistence
            st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# This ensures the script runs properly when executed directly
# ==============================================================================

if __name__ == "__main__":
    main()