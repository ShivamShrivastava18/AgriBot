import pinecone
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers


#Extract data from PDF
def load_pdf(data):
    loader = DirectoryLoader(data,  
                    glob ="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
extracted_data = load_pdf("Data/")

#Creating Text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#Download embeddings
def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings = download_huggingface_embeddings()