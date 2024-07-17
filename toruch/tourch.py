import pdfplumber
import torch
from sentence_transformers import SentenceTransformer, util

# Load a lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def find_information(text, query):
    sentences = text.split('\n')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return sentences[best_idx]

def extract_invoice_details(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    
    seller_name_query = "seller name"
    total_amount_query = "total amount"
    
    seller_name = find_information(text, seller_name_query)
    total_amount = find_information(text, total_amount_query)
    
    return seller_name, total_amount

# Path to your PDF file
pdf_path = '/mnt/data/Capture.PNG'

# Extract details
seller_name, total_amount = extract_invoice_details(pdf_path)

print("Seller Name:", seller_name)
print("Total Amount:", total_amount)
