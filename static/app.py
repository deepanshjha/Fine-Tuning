from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, ConversationChain
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="deep"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)

smriti = ConversationBufferWindowMemory(k=2,memory_key="history")

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':1024,
                          'temperature':0.8})

_DEFAULT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer and don't repeat any answer.

{history}

Question: {input}
Only return the helpful answer below and nothing else.
Helpful answer:"""
PROMPT = PromptTemplate(
   input_variables=["history","input"], template=_DEFAULT_TEMPLATE
)
qa = ConversationChain(
   llm=llm,
   prompt=PROMPT,
   memory=smriti,
   verbose=True
)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"input": input})
    print("Response : ", result["response"])
    return str(result["response"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


