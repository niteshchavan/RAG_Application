from flask import Flask, request, render_template, jsonify
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Define the upload folder

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

folder_path = 'db2'
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

@app.route('/')
def index():
    return render_template('retriver.html')

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
    print(f"File saved to: {filepath}")

    try:
        loader = PDFPlumberLoader(filepath)
        docs = loader.load_and_split()
        
        chunks = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )    
        vector_store.persist()
        return jsonify({'message': 'File processed and stored successfully'}), 200
    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        return jsonify({'message': 'An error occurred during document processing'}), 500

try:
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
except Exception as e:
    print(f"An error occurred while loading the vector store: {e}")
    vector_store = None

if vector_store:
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

@app.route('/query', methods=['POST'])
def query():
    if vector_store is None:
        return jsonify({'message': 'Vector store is not available'}), 500

    try:
        data = request.get_json()
        query_text = data.get("query_text", "")
        print(query_text)
        relevant_documents = retriever.invoke(query_text)
        results = [doc.page_content for doc in relevant_documents]
        print(results)
        #return jsonify(results), 200
        return jsonify({'results': results}), 200
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return jsonify({'message': 'An error occurred during retrieval'}), 500

if __name__ == '__main__':
    app.run(debug=True)
