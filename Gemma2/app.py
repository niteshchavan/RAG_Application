import os
import json
from flask import Flask, request, render_template, jsonify
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate



app = Flask(__name__)

llm = Ollama(model="gemma2")

CHAT_HISTORY_FILE = 'chat_history.json'

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

embedding = FastEmbedEmbeddings()

folder_path = 'db'

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Load existing chat history if file exists
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            chat_history = json.load(f)
    except json.JSONDecodeError:
        chat_history = []
else:
    chat_history = []

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI named Nitesh, you answer questions with simple answers and no funny stuff."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


raw_prompt = PromptTemplate.from_template(""" 
    <s>[INST] You are a technical assistant good at searching documents. Keep your answers concise and limit your response to 100 words. If you do not have an answer from the provided information, say so. [/INST]</s>

        [INST]{input}
            Context: {context}
            Answer:
        [/INST]                                
""")

chain = prompt_template | llm

try:
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
except Exception as e:
    print(f"An error occurred: {e}")

@app.route('/')
def index():
    return render_template('index.html')

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.1,
    },
)    
    
document_chain = create_stuff_documents_chain(llm, raw_prompt)
chain = create_retrieval_chain(retriever, document_chain)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    
    
    

    
    # Generate response using the chat prompt template and LLM
    response = chain.invoke({"input": user_input})


    try:
        # Append messages to chat history after converting to JSON-serializable format
        chat_history.append({
            "role": "human",
            "content": user_input.__dict__ if isinstance(user_input, HumanMessage) else user_input
        })
        chat_history.append({
            "role": "ai",
            "content": response.__dict__ if isinstance(response, AIMessage) else response
        })

        # Save chat history to file
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(chat_history, f, default=str)  # Use default=str to handle non-serializable objects

    except IOError as e:
        print(f"Error saving file: {e}")


    # Extracting just the 'answer' part
    answer_data = {
        'answer': response['answer']
    }

    return jsonify(answer_data)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400
    
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    loader = PDFPlumberLoader(filepath)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )    
    vector_store.persist()
    
    return jsonify({'message': f'File {filename} uploaded successfully'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
