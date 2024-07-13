import chromadb

folder_path = 'db'

chroma_client = chromadb.PersistentClient(path=folder_path)

try:
    # Attempt to create the collection
    collection = chroma_client.create_collection(name="my_collection1")
except chromadb.db.base.UniqueConstraintError:
    # Collection already exists, so retrieve it instead
    collection = chroma_client.get_collection(name="my_collection1")

# Add documents without specifying IDs (let chromadb generate them)
documents = [
    {"content": "This is a document about pineapple"},
    {"content": "This is a document about oranges"},
    {"content": "Date is 12 July 2024"}
]
collection.add(documents=documents)

# Uncomment to query documents from the collection
results = collection.query(
    query_texts=["This is a query document about"],  # Chroma will embed this for you
    #limit=10  # Limiting the number of results to return
)

print(results)
