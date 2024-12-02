from flask import Flask, render_template, jsonify, request
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from src.prompt import prompt_template

from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import pinecone  # Ensure you have pinecone installed

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Loading the index
index_name="support"
# docsearch = Pinecone.from_existing_index(index_name, embeddings)
docsearch=Pinecone.from_existing_index(index_name, embeddings)


# Use the prompt template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["Question ", "Answer"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

# Chain Type Arguments - including 'document_variable_name' to specify the variable name for the context
chain_type_kwargs = {
    "document_variable_name": "context",  # Ensuring the context variable name is properly set
}

# Initialize the RetrievalQA chain with the correct setup
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs  # Pass the correct configuration for context
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)