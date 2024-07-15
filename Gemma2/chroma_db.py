import os
import uuid
import chromadb
from flask import Flask, request, render_template, jsonify
from chromadb.db.base import UniqueConstraintError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

@app.route('/')
def index():
    return render_template('index.html')

folder_path = 'db'
chroma_client = chromadb.PersistentClient(path=folder_path)

try:
    # Attempt to create the collection
    collection = chroma_client.create_collection(name="my_collection1")
except UniqueConstraintError:
    # Collection already exists, so retrieve it instead
    collection = chroma_client.get_collection(name="my_collection1")

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
        
        # Ensure chunks are strings
        chunks = text_splitter.split_documents(docs)
        chunk_texts = [chunk.page_content for chunk in chunks]
        print(f"Chunks: {chunk_texts}")
        
        # Generate unique IDs for each document chunk
        ids = [str(uuid.uuid4()) for _ in chunk_texts]
        
        collection.add(documents=chunk_texts, ids=ids, metadatas=[{"source": filename}] * len(chunk_texts))
                # Check if embeddings are generated

        added_documents = collection.get(ids=ids)

        print(f"Added documents: {added_documents}")
        
        return jsonify({'message': 'File processed and data added to collection successfully'})
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'message': f'Error processing file: {e}'}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query_text", "")
        n_results = data.get("n_results", 2)
        
        if not query_text:
            return jsonify({'message': 'Query text is required'}), 400
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return jsonify(results)
    except Exception as e:
        print(f"Error querying the collection: {e}")
        return jsonify({'message': f'Error querying the collection: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
