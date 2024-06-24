from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint


app = Flask(__name__)

load_dotenv()

HF_TOKEN = os.environ.get('HF_TOKEN')

from huggingface_hub import login

login(token=HF_TOKEN) 

ASTRA_DB_API_ENDPOINT= os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE = "RAG"
TABLE_NAME = 'DJ1'


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=TABLE_NAME,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_TOKEN,
    namespace=ASTRA_DB_KEYSPACE,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an astrology assistant Deepansh Jha for question-answering astrology tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

os.environ["HUGGINGFACEHUB_API_TOKEN"]= HF_TOKEN
llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", api_key=HF_TOKEN)

# llm = ChatOllama(model="deepudj/dj-phi3", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("Question : ", input)
    docs = retriever.invoke(input)
    formatted_docs = format_docs(docs)  # Use the format_docs function to format the documents
    result = rag_chain.invoke({"context": formatted_docs, "question": input})
    print("Response : ", result)
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


