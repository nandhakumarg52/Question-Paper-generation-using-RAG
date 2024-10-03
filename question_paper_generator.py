import os
import warnings
from uuid import uuid4
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Optional, Any, Type

# Ignore warnings for clean output
warnings.filterwarnings("ignore")
load_dotenv()

# --- Document Processor ---
class DocumentProcessor:
    def __init__(self, file_path, chunk_size=512, chunk_overlap=50):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_documents(self):
        """Load and split the document into chunks."""
        loader = PyPDFLoader(self.file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return loader.load_and_split(text_splitter)

# --- Vector Store Manager ---
class VectorStoreManager:
    def __init__(self, qdrant_url, qdrant_api_key, hf_token, embedding_model_name="hf", model_name="sentence-transformers/all-roberta-large-v1"):
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=100)

        # Initialize the correct embedding model based on embedding_model_name
        if embedding_model_name == "hf":
            login(token=hf_token)
            self.embedding = self.get_huggingface_embedding(model_name=model_name)
        elif embedding_model_name == "ihf":
            login(token=hf_token)
            self.embedding = self.get_huggingface_inference_embedding(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding model name: {embedding_model_name}")

    def get_huggingface_embedding(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Initialize Hugging Face embedding
        return HuggingFaceEmbeddings(model_name=model_name)

    def get_huggingface_inference_embedding(self, model_name):
        # Initialize Hugging Face inference embedding
        # Placeholder for actual embedding logic
        pass

    def process_file(self, file_path):
        """Process the file by checking for collection existence and inserting vectors if new."""
        self.collection_name = self._generate_collection_name(file_path)
        
        if not self._collection_exists():
            print(f"Collection '{self.collection_name}' does not exist. Creating a new collection and upserting documents.")
            self.create_collection(vector_size=1024, distance=Distance.COSINE)

            # Process and load the document
            doc_processor = DocumentProcessor(file_path)
            loaded_documents = doc_processor.load_and_split_documents()

            # Upsert the documents to the collection
            self.upsert_documents(loaded_documents)
        else:
            print(f"Collection '{self.collection_name}' already exists. Using the existing collection.")

        return self.initialize_vectorstore()

    def _generate_collection_name(self, file_path):
        """Generate a unique collection name based on the file name."""
        return os.path.splitext(os.path.basename(file_path))[0]

    def _collection_exists(self):
        """Check if the collection already exists."""
        collections = self.qdrant_client.get_collections().collections
        existing_collections = [collection.name for collection in collections]
        return self.collection_name in existing_collections

    def create_collection(self, vector_size=1024, distance=Distance.COSINE):
        """Create a new collection in Qdrant."""
        print(f"Creating new collection '{self.collection_name}'.")
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "content": VectorParams(size=vector_size, distance=distance)
            }
        )

    def upsert_documents(self, documents):
        """Upsert document vectors into Qdrant."""
        chunked_metadata = []
        for item in documents:
            id = str(uuid4())
            content = item.page_content
            source = item.metadata.get("source", "")
            page = item.metadata.get("page", "")
            content_vector = self.embedding.embed_documents([content])[0]
            vector_dict = {"content": content_vector}
            payload = {
                "page_content": content,
                "metadata": {
                    "id": id,
                    "page_content": content,
                    "source": source,
                    "page": page,
                }
            }
            metadata = PointStruct(id=id, vector=vector_dict, payload=payload)
            chunked_metadata.append(metadata)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=chunked_metadata
        )

    def initialize_vectorstore(self):
        """Initialize the vector store for querying."""
        return QdrantVectorStore(client=self.qdrant_client,
                                 collection_name=self.collection_name,
                                 embedding=self.embedding,
                                 vector_name="content")

# --- QA ChatBot ---
class QAChatBot:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.qa_chain = self._initialize_qa_chain()

    def _initialize_qa_chain(self):
        template = """
# System Prompt for Generating Question Papers
You are an AI system designed to generate examination question papers...
(Include the full instructions for question paper generation here)
        """
        retriever = self.vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_template(template)
        handler = StdOutCallbackHandler()

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            callbacks=[handler],
            retriever=retriever
        )

    def run(self, question: str) -> str:
        if not question:
            return "No question provided."
        try:
            result = self.qa_chain.invoke(question)
            return result or "Failed to retrieve an answer."
        except Exception as e:
            return f"Error during question processing: {e}"

# --- Retriever Tool ---
class RetrieverToolSchema(BaseModel):
    question: str = Field(
        ..., description="The question to retrieve an answer for using the QAChatBot."
    )

class RetrieverTool(BaseTool):
    name: str = "Textbook Retriever"
    description: str = "Elementary school textbook retriever"
    args_schema: Type[BaseModel] = RetrieverToolSchema
    qa_chatbot: Optional[Any] = None

    def __init__(self, qa_chatbot: Optional[Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.qa_chatbot = qa_chatbot

    def _run(self, **kwargs: Any) -> Any:
        question = kwargs.get("question")
        if not question:
            return "No question provided."
        return self.qa_chatbot.run(question)

# --- Question Paper Generator (QGen) ---
class QGen:
    def __init__(self, vectorstore, llm, question_pattern=None):
        self.llm = llm
        self.qa_chatbot = QAChatBot(vectorstore=vectorstore, llm=self.llm)
        self.rag_tool = RetrieverTool(qa_chatbot=self.qa_chatbot)
        self.question = question_pattern

    def generate_question_paper(self):
        inputs = {"question": self.question}
        return self.rag_tool._run(**inputs)["result"]
