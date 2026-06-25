# rag.py
import os
import re
import fitz  # PyMuPDF
import numpy as np
import hashlib
import pickle
import logging
from PIL import Image
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
from transformers import CLIPProcessor, CLIPModel
from google import genai
from google.genai import types as genai_types
import nltk
import easyocr
import psutil
from dotenv import load_dotenv

load_dotenv()

# ── Global state ─────────────────────────────────────────────────────────────
processed_data = None
chat_history   = []

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── NLTK data ────────────────────────────────────────────────────────────────
for pkg in ('punkt', 'punkt_tab'):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Embedding & vision models ─────────────────────────────────────────────────
logger.info("Loading SentenceTransformer…")
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='./model_cache')

logger.info("Loading CLIP…")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./model_cache')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir='./model_cache')

# ── Google Gemini client ──────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set in your .env file.")

gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL  = "gemma-4-31b-it"
logger.info("Gemini client ready (%s).", GEMINI_MODEL)


# ── PDF Processing ────────────────────────────────────────────────────────────

def _process_page_sequential(page_num, pdf_path, max_images, image_counter_start):
    """Extract text + images from a single PDF page (runs in the main process)."""
    content        = []
    image_metadata = {}
    image_counter  = image_counter_start

    try:
        doc  = fitz.open(pdf_path)
        page = doc.load_page(page_num)

        content.append(page.get_text() + "\n")

        if max_images == -1 or image_counter_start < max_images:
            for img in page.get_images(full=True):
                if max_images != -1 and image_counter >= max_images:
                    break
                xref = img[0]
                try:
                    base_image  = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext     = base_image["ext"]
                    img_path    = f"figures/figure-{image_counter}.{img_ext}"

                    os.makedirs("figures", exist_ok=True)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                    img_key = f"Image {image_counter}"
                    content.append(f"[{img_key}]\n")
                    image_metadata[img_key] = {
                        "path":        img_path,
                        "description": "OCR deferred to query time",
                    }
                    image_counter += 1
                except Exception as e:
                    logger.warning(f"Image extraction failed on page {page_num}: {e}")

        doc.close()
    except Exception as e:
        logger.error(f"Error processing page {page_num}: {e}")

    return "".join(content), image_metadata, image_counter


def extract_content_from_pdf(pdf_path, max_images=50):
    """Extract text and images from a PDF sequentially (safe on all platforms)."""
    os.makedirs("figures", exist_ok=True)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        num_pages = len(doc)

    content        = []
    image_metadata = {}
    image_counter  = 1

    for page_num in range(num_pages):
        page_content, page_meta, image_counter = _process_page_sequential(
            page_num, pdf_path, max_images, image_counter
        )
        if page_content.strip():
            content.append(page_content)
        image_metadata.update(page_meta)

    return "".join(content).strip(), image_metadata


# ── Chunking & Indexing ───────────────────────────────────────────────────────

def recursive_chunking(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separator='\n',
    )
    return splitter.split_text(text)


def build_faiss_index(embeddings):
    """Build a flat L2 FAISS index from embeddings."""
    dim   = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index


# ── Retrieval ────────────────────────────────────────────────────────────────

def hybrid_search(query, index, chunks, image_metadata, alpha=0.5, top_k=5):
    """Hybrid BM25 + vector search."""
    if not chunks or index is None:
        logger.error("hybrid_search: empty chunks or missing index.")
        return []

    # ── Vector scores ──
    q_emb = embed_model.encode(query, normalize_embeddings=True)
    q_emb = np.array([q_emb], dtype='float32')
    distances, _ = index.search(q_emb, len(chunks))
    vector_scores = [1 / (1 + d) for d in distances[0]]

    # ── BM25 scores ──
    tokenized = [word_tokenize(c.lower()) for c in chunks]
    bm25      = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(word_tokenize(query.lower()))

    # ── Combine & rank ──
    combined = sorted(
        [(i, alpha * bm25_scores[i] + (1 - alpha) * vector_scores[i])
         for i in range(len(chunks))],
        key=lambda x: x[1],
        reverse=True,
    )
    results = [chunks[i] for i, _ in combined[:top_k]]

    # ── Attach image metadata if user referenced an image ──
    img_ref = re.search(r"Image (\d+)", query, re.IGNORECASE)
    if img_ref:
        img_key = f"Image {img_ref.group(1)}"
        if img_key in image_metadata:
            results.append(f"[Image Metadata]: {image_metadata[img_key]['description']}")

    return results


# ── CLIP image analysis ───────────────────────────────────────────────────────

def explain_image_with_clip(query, image_path, img_key, image_metadata):
    """Run on-demand OCR + CLIP similarity for an image."""
    try:
        image = Image.open(image_path).convert("RGB")

        if image_metadata[img_key]["description"] == "OCR deferred to query time":
            reader      = easyocr.Reader(['en'], gpu=False)
            ocr_result  = reader.readtext(image_path, detail=0)
            description = " ".join(ocr_result) if ocr_result else "No OCR text found"
            image_metadata[img_key]["description"] = description
        else:
            description = image_metadata[img_key]["description"]

        inputs  = clip_processor(text=[query], images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        sim     = outputs.logits_per_image.item()
        return f"Image relevance score: {sim:.2f}. OCR text: {description}"
    except Exception as e:
        logger.error(f"CLIP analysis failed: {e}")
        return "Image analysis failed."


# ── PDF caching ───────────────────────────────────────────────────────────────

def _pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_or_process_pdf(pdf_path):
    """Process a PDF (or load from cache) and return the RAG data tuple."""
    cache_dir  = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{_pdf_hash(pdf_path)}.pkl")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached data for {pdf_path}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(f"Processing PDF: {pdf_path}")
    text, image_metadata = extract_content_from_pdf(pdf_path, max_images=50)

    # Guard: fail early if there is no extractable text
    if not text.strip():
        raise ValueError("PDF contains no extractable text. It may be a scanned image-only PDF.")

    chunks     = recursive_chunking(text, chunk_size=500, overlap=50)
    embeddings = embed_model.encode(
        chunks,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index = build_faiss_index(embeddings)

    with open(cache_file, "wb") as f:
        pickle.dump((text, image_metadata, chunks, embeddings, index), f)

    logger.info(f"PDF processed: {len(chunks)} chunks, {len(image_metadata)} images.")
    return text, image_metadata, chunks, embeddings, index


# ── RAG Query ────────────────────────────────────────────────────────────────

def query_rag(user_query, index, chunks, image_metadata, chat_history):
    """Run hybrid retrieval then call Gemini to generate an answer."""
    try:
        retrieved = hybrid_search(user_query, index, chunks, image_metadata)

        # On-demand image analysis
        img_ref = re.search(r"Image (\d+)", user_query, re.IGNORECASE)
        if img_ref:
            img_key = f"Image {img_ref.group(1)}"
            if img_key in image_metadata:
                explanation = explain_image_with_clip(
                    user_query,
                    image_metadata[img_key]["path"],
                    img_key,
                    image_metadata,
                )
                retrieved.append(f"[Image Analysis]: {explanation}")

        # Build recent history string (last 6 turns to stay within context)
        history_str = ""
        for turn in chat_history[-6:]:
            history_str += f"User: {turn['query']}\nAssistant: {turn['response']}\n\n"

        prompt = f"""You are an expert assistant. Answer the user's question using the provided context.

## Rules
- Be natural and conversational — don't be robotic or overly formal.
- If the question is simple, give a concise answer. If it needs explanation, elaborate freely.
- Never mention that you're using "context" or "retrieved chunks" — just answer.
- Use markdown formatting: headings, bullet points, code blocks where appropriate.
- If you need to reason beyond the context, do so silently.

## Recent Conversation
{history_str if history_str else "No prior conversation."}

## Retrieved Document Context
{chr(10).join(retrieved) if retrieved else "No relevant context found."}

## User Question
{user_query}
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=1600,
                temperature=0.7,
            ),
        )

        # Safely extract text
        answer = None
        try:
            answer = response.text
        except Exception:
            pass

        if not answer:
            try:
                answer = response.candidates[0].content.parts[0].text
            except Exception:
                pass

        if not answer:
            logger.warning("Gemini returned empty content. Response: %s", response)
            answer = "The model returned an empty response. Please try rephrasing your question."

        chat_history.append({"query": user_query, "response": answer})
        return answer

    except Exception as e:
        logger.error(f"query_rag failed: {e}", exc_info=True)
        return f"An error occurred while generating the response: {str(e)}"


# ── Public API (called by app.py) ─────────────────────────────────────────────

def process_uploaded_pdf(pdf_path):
    global processed_data, chat_history
    processed_data = load_or_process_pdf(pdf_path)
    chat_history   = []
    return {"message": "PDF processed successfully"}


def handle_query(user_query):
    global processed_data, chat_history
    if processed_data is None:
        return {"error": "No PDF processed. Please upload a document first."}

    _, image_metadata, chunks, _, index = processed_data
    answer = query_rag(user_query, index, chunks, image_metadata, chat_history)
    return {"response": answer, "chat_history": chat_history[-10:]}