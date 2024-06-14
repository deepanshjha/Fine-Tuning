from langchain.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


#Extract data from the PDF
def load_pdf(data):
    loader1 = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    document1 = loader1.load()

    loader2 = DirectoryLoader(data,
                    glob="*.docx",
                    loader_cls=Docx2txtLoader)
    
    document2 = loader2.load()

    document = document1+document2

    return document