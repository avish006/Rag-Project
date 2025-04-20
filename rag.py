# rag.py
import os
import re
import fitz  # PyMuPDF
import pytesseract
import numpy as np
import hashlib
import pickle
import logging
from PIL import Image
from io import BytesIO
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import nltk
from multiprocessing import Pool
from functools import partial
import easyocr
import psutil
from dotenv import load_dotenv
import os

load_dotenv()



# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')

# Initialize models
model = SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder = './model_cache')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './model_cache')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",cache_dir = './model_cache')


# Initialize OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))

def process_page(args):
    """Process a single page for text and images (no OCR during initial load)"""
    page_num, pdf_path, max_images, image_counter_start = args
    content = []
    image_metadata = {}
    image_counter = image_counter_start

    # Log memory usage
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    logger.debug(f"Page {page_num}: Memory before: {mem_before:.2f} MB")

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # Extract text
        content.append(page.get_text() + "\n")

        # Extract images (if within limit), but skip OCR
        if max_images == -1 or image_counter_start < max_images:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                if max_images != -1 and image_counter >= max_images:
                    break
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image['ext']
                    img_path = f"figures/figure-{image_counter}.{img_ext}"
                    
                    # Save image
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    img_key = f"Image {image_counter}"
                    content.append(f"[{img_key}]\n")
                    image_metadata[img_key] = {
                        "path": img_path,
                        "description": "OCR deferred to query time"
                    }
                    image_counter += 1
                except Exception as e:
                    logger.error(f"Image extraction failed on page {page_num}: {str(e)}")

        doc.close()
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {str(e)}")

    # Log memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    logger.debug(f"Page {page_num}: Memory after: {mem_after:.2f} MB, Delta: {mem_after - mem_before:.2f} MB")

    return "".join(content), image_metadata, image_counter

def extract_content_from_pdf(pdf_path, max_images=50, max_processes=None):
    """Extract text and images from PDF using PyMuPDF (no OCR initially)"""
    os.makedirs("figures", exist_ok=True)
    content = []
    image_metadata = {}
    image_counter = 1

    # Validate PDF path
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Get number of pages
    with fitz.open(pdf_path) as doc:
        num_pages = len(doc)

    # Prepare arguments for parallel processing
    args = [(i, pdf_path, max_images, image_counter + i * 10) for i in range(num_pages)]

    # Determine number of processes (default to CPU count, but cap if specified)
    if max_processes is None:
        max_processes = os.cpu_count() or 1
    max_processes = min(max_processes, os.cpu_count() or 1)

    # Parallel processing with limited processes
    try:
        with Pool(processes=max_processes) as pool:
            results = pool.map(process_page, args)
    except Exception as e:
        logger.error(f"Multiprocessing failed: {str(e)}")
        raise

    # Aggregate results
    for page_content, page_image_metadata, _ in results:
        if page_content:
            content.append(page_content)
        image_metadata.update(page_image_metadata)

    return "".join(content).strip(), image_metadata



def recursive_chunking(text, chunk_size=500, overlap=100):
    """Split text into chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator= '\n'
    )
    return text_splitter.split_text(text)

def store_embeddings_faiss(embeddings):
    """Create FAISS index"""
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    np_embeddings = np.array(embeddings).astype('float32')
    index.add(np_embeddings)
    return index

def hybrid_search(query, index, chunks, image_metadata, alpha=0.5, top_k=5):
    """Hybrid vector + BM25 search"""
    # Vector search
    if not chunks or index is None:
        logger.error("Invalid search inputs - empty chunks or missing index")
        return []
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, len(chunks))
    vector_scores = [1 / (1 + d) for d in distances[0]]

    # BM25 search
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Combine scores
    combined = []
    for i in range(len(chunks)):
        score = alpha * bm25_scores[i] + (1 - alpha) * vector_scores[i]
        combined.append((i, score))
    
    # Get top results
    combined.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in combined[:top_k]]
    results = [chunks[i] for i in top_indices]

    # Add image metadata if referenced
    image_ref = re.search(r"Image (\d+)", query, re.IGNORECASE)
    if image_ref and (img_key := f"Image {image_ref.group(1)}") in image_metadata:
        results.append(f"[Image Metadata]: {image_metadata[img_key]['description']}")

    return results

def explain_image_with_clip(query, image_path, img_key, image_metadata):
    """CLIP-based image explanation with on-demand OCR"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Perform OCR if deferred
        if image_metadata[img_key]["description"] == "OCR deferred to query time":
            reader = easyocr.Reader(['en'], gpu=False)  # gpu=True if available
            description = reader.readtext(image_path, detail=0)
            description = " ".join(description) if description else "No OCR text available"
            image_metadata[img_key]["description"] = description
        else:
            description = image_metadata[img_key]["description"]

        inputs = clip_processor(
            text=[query], 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        outputs = clip_model(**inputs)
        similarity = outputs.logits_per_image.item()
        return f"Image relevance: {similarity:.2f}. OCR text: {description}"
    except Exception as e:
        logger.error(f"CLIP error: {str(e)}")
        return "Image analysis failed"

def get_pdf_hash(pdf_path):
    """Generate PDF hash for caching"""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_or_process_pdf(pdf_path):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    pdf_hash = get_pdf_hash(pdf_path)
    cache_file = f"{cache_dir}/{pdf_hash}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    text, image_metadata = extract_content_from_pdf(pdf_path, max_images=50, max_processes=2)
    chunks = recursive_chunking(text, chunk_size=500, overlap=50)
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='./model_cache')
    embeddings = model.encode(chunks, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    index = store_embeddings_faiss(embeddings)

    if not text.strip():
        raise ValueError("PDF contains no extractable text")

    with open(cache_file, "wb") as f:
        pickle.dump((text, image_metadata, chunks, embeddings, index), f)

    return text, image_metadata, chunks, embeddings, index

def query_rag(user_query, index, chunks, image_metadata, chat_history):
    """Handle RAG query"""
    try:

        retrieved = hybrid_search(user_query, index, chunks, image_metadata)
        
        # Image handling
        if image_ref := re.search(r"Image (\d+)", user_query, re.IGNORECASE):
            img_key = f"Image {image_ref.group(1)}"
            if img_key in image_metadata:
                img_path = image_metadata[img_key]["path"]
                img_explanation = explain_image_with_clip(user_query, img_path, img_key, image_metadata)
                retrieved.append(f"[Image Analysis]: {img_explanation}")
        
        prompt = f"""Query: {user_query}
        Context: {retrieved}
        Chat History: {chat_history}
        You are an expert assistant proficient in maths, logic, explaination of complex concepts , When asked a direct question provide a simple straightforward response don't yap unneccesarily (but also don't shorten to one word answer keep it natural like human) , However if task require explaination or reasoning feel free to yap
        Don't Talk about the context just provide the required response based on the query , Context , but you are allowed to perform logical reasoning if required.
        The context you have been given is the "Context" and "Chat History" If user's query involves looking into the previous queries or responses use the context of Chat History and your logic

        Again Don't tell if you are using your own context or you are using your own reasoning , also respond in formatted way with appropriate headings or subheadings and make it look clean and readable.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1600,
            temperature=0.7
        )
        chat_history.append({"query": user_query, "response": response.choices[0].message.content})
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return f"Error processing query: {str(e)}"

def process_uploaded_pdf(pdf_path):
    global processed_data, chat_history
    processed_data = load_or_process_pdf(pdf_path)
    chat_history = []
    return {"message": "PDF processed successfully"}

def handle_query(user_query):
    global processed_data, chat_history
    if not processed_data:
        return {"error": "No PDF processed"}
    
    _, image_metadata, chunks, _, index = processed_data
    response = query_rag(user_query, index, chunks, image_metadata, chat_history)
    
    return {"response": response, "chat_history": chat_history[-10:]}