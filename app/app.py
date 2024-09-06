from flask import Flask, render_template, request, redirect, url_for
import fitz  # PyMuPDF for PDF text extraction
import re
import evaluate
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load your local Word2Vec model
model_word2vec_indo_path = 'models\word2vec-indonesia.model'  # Update this path to your local model
model_word2vec_indo = Word2Vec.load(model_word2vec_indo_path)

# Load the ROUGE evaluation metric
rouge = evaluate.load('rouge')

# Function to clean and process text
def clean_text(text):
    text = re.sub("@[A-Za-z0-9]+", " ", text)
    text = re.sub("#[A-Za-z0-9_]+", " ", text)
    text = re.sub('https:\/\/\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract information using regex patterns
def extract_information(text):
    patterns = {
    "Nama": r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b",
    "Tanggal": r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b(?![-\d])|\b\d{1,2}\s(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Telepon": r"(?:(?:\+62|0)[-\s]?\d{2,4})[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{2,4})?\b(?!\.\d)",
    "URL": r"\bhttps?:\/\/[^\s/$.?#].[^\s]*\b",
    "Kata Benda": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
    "Akronim": r"\b([A-Z]{2,})\b",
    "Nilai Mata Uang": r"(?:Rp|USD|\$|EUR|£|€)\s?\d{1,3}(?:[.,]\d{3})*(?:\s?(?:ribu|juta|miliar|triliun))?(?:[.,]\d{2})?|\b\d{1,3}(?:[.,]\d{3})*(?:\s?(?:ribu|juta|miliar|triliun))?\s?(?:Rp|USD|\$|EUR|£|€)\b",
    "Alamat IP": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "Kode Pos": r"\b\d{5}(?:-\d{4})?\b"
}


    results = {}

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            results[key] = list(set(matches))  # Remove duplicates

    return results

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file):
    extracted_text = ''
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            extracted_text += page.get_text("text")  # Extract text from each page
    return extracted_text

# Embedder function to get the sentence embedding using the Word2Vec model
def embed_sentence(model, sentence, return_oov=False):
    tokens = re.findall(r"\w+", sentence.lower())
    list_vec = []
    OOV_tokens = []
    
    for token in tokens:
        try:
            vec = model.wv.get_vector(token)
            list_vec.append(vec)
        except KeyError:
            OOV_tokens.append(token)

    if not list_vec:
        return (False, OOV_tokens) if return_oov else False

    return (np.mean(list_vec, axis=0), OOV_tokens) if return_oov else np.mean(list_vec, axis=0)

# Function to summarize the text using clustering
def summarize_text(text, model, ratio=0.2, num_sentences=None, use_first=True):
    sentences = sent_tokenize(text)
    embeddings = []
    
    # Create embeddings for each sentence
    for sentence in sentences:
        embedding = embed_sentence(model, sentence)
        if isinstance(embedding, np.ndarray):
            embeddings.append(embedding)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)

    # Define number of clusters (k)
    if num_sentences is not None:
        k = min(num_sentences, len(embeddings))
    else:
        k = max(int(len(embeddings) * ratio), 1)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(embeddings)

    # Get cluster centers
    centroids = kmeans.cluster_centers_

    # Find the sentences closest to each cluster center
    summary_idx = find_closest_sentences(embeddings, centroids)

    if use_first and 0 not in summary_idx:
        summary_idx.insert(0, 0)

    # Return the summary sentences
    summary = [sentences[idx] for idx in sorted(summary_idx)]
    return ' '.join(summary)

# Function to find the closest sentences to cluster centers
def find_closest_sentences(embeddings, centroids):
    closest_sentences = []
    used_indices = set()

    for centroid in centroids:
        closest_idx = None
        closest_dist = float('inf')
        
        for i, embedding in enumerate(embeddings):
            if i not in used_indices:
                dist = np.linalg.norm(embedding - centroid)
                if dist < closest_dist:
                    closest_idx = i
                    closest_dist = dist
        
        used_indices.add(closest_idx)
        closest_sentences.append(closest_idx)

    return closest_sentences

# Function to calculate ROUGE score between original text and generated summary
def calculate_rouge(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('pdf')
        input_text = request.form.get('input_text')

        if file and file.filename.endswith('.pdf'):
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(file)
        elif input_text:
            # Use the provided text input
            extracted_text = input_text
        else:
            extracted_text = ''

        if extracted_text:
            # Extract information before any preprocessing
            extracted_info = extract_information(extracted_text)

            # Clean the text
            cleaned_text = clean_text(extracted_text)

            # Summarize the cleaned text
            summary = summarize_text(cleaned_text, model_word2vec_indo)
            
            # Calculate ROUGE score between the original extracted text and the generated summary
            rouge_results = calculate_rouge([summary], [extracted_text])

            # Pass all extracted information, full text, and summary to the template
            return render_template('index.html', summary=summary, text=cleaned_text, full_text=extracted_text, extracted_info=extracted_info, rouge=rouge_results)

    return render_template('index.html', summary='', text='', full_text='', extracted_info={}, rouge={})

# Route to handle clearing the data
@app.route('/clear', methods=['GET'])
def clear():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
