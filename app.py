from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from rag import process_uploaded_pdf, handle_query
import os
import tempfile
import logging
import pytesseract
import time

# === Config ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH_MB = 50
TESSERACT_CMD = os.getenv('TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        safe_name = f"uploaded_{int(time.time())}.pdf"
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(save_path)
        
        result = process_uploaded_pdf(save_path)
        pdf_url = f"/uploaded/{safe_name}"
        return jsonify({"message": "PDF processed!", "pdf_url": pdf_url}), 200
    
    except Exception as e:
        return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        result = handle_query(data['query'].strip())
        return jsonify({"response": result.get('response', '')}), 200
    
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

@app.route('/uploaded/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    app.run(host='0.0.0.0', port=5000, debug=False)