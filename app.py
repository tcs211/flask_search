from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import json
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import jieba
import re

import ssl
import sys
import string


app = Flask(__name__)
# print(sys.path)
root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
CORS(app, resources={r"/*": 
                     {"origins": "*",
                    "methods": ["GET", "POST"],
                    "allow_headers": ["Content-Type", "Authorization"]}})  # You can replace "*" with specific domains

app.config['UPLOAD_FOLDER'] = os.path.join(root_path, 'public', 'documents')

# set the path to the nltk_data folder if system is linux
if os.name == 'posix':
    nltk.data.path.append('/home/www/nltk_data')
    # create nltk_data folder if it doesn't exist
    os.makedirs('/home/www/nltk_data', exist_ok=True)
# create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download necessary NLTK data
nltk.download('punkt', download_dir=nltk.data.path[0])
nltk.download('punkt_tab', download_dir=nltk.data.path[0])
nltk.download('stopwords', download_dir=nltk.data.path[0])

# Initialize NLTK components
porter_stemmer = PorterStemmer()
english_stop_words = set(stopwords.words('english'))

# Load Chinese stop words (you may need to provide your own list)
with open(os.path.join(root_path, 'chinese_stop_words.txt'), 'r', encoding='utf-8') as f:
    chinese_stop_words = set(f.read().splitlines())

# Initialize Jieba

jieba.set_dictionary(os.path.join(root_path, 'dict.txt.big'))
    # './dict.txt.big'

# Custom index structure
index = {
    'inverted_index': {},
    'document_store': {}
}

DB_FILE = os.path.join(root_path, 'local_database.json')
# './local_database.json'

def load_index():
    global index
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
            if not data:
                index = {'inverted_index': {}, 'document_store': {}}
                return
            index = json.loads(data)
            if 'inverted_index' not in index:
                index['inverted_index'] = {}
            if 'document_store' not in index:
                index['document_store'] = {}
        print('Loaded existing index from file')
    except FileNotFoundError:
        print('No existing index found, starting with an empty index')
        index = {'inverted_index': {}, 'document_store': {}}
    except json.JSONDecodeError:
        print('Error decoding JSON, starting with an empty index')
        index = {'inverted_index': {}, 'document_store': {}}

# Load existing index on startup
load_index()

def save_index():
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False)
        print('Index saved to file')
    except Exception as e:
        print(f'Error saving index: {e}')

def detect_language(text):
    # Remove spaces and punctuation
    text = ''.join(char for char in text if char not in string.punctuation and not char.isspace())
    
    # Counter for Chinese characters
    chinese_char_count = 0
    simplified_char_count = 0
    traditional_char_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_char_count += 1
            # Check for simplified Chinese characters
            if '\u4e00' <= char <= '\u9fa5':
                simplified_char_count += 1
            # Check for traditional Chinese characters
            if '\u9fa6' <= char <= '\u9fff':
                traditional_char_count += 1
    
    # If more than 20% of characters are Chinese, consider it Chinese
    if chinese_char_count / len(text) > 0.2:
        if traditional_char_count > simplified_char_count:
            return 'zh-tw'
        else:
            return 'zh-cn'
    else:
        return 'en'


def tokenize_and_stem(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    language = detect_language(text)
    print ('language:', language)
    if language == 'zh-cn' or language == 'zh-tw':
        # Use jieba for Chinese tokenization
        tokens = jieba.cut_for_search(text)
        print ('tokens:', tokens)
        return [token for token in tokens if token not in chinese_stop_words and token.strip()]
    else:
        tokens = word_tokenize(text.lower())
        return [porter_stemmer.stem(token) for token in tokens if token not in english_stop_words]

def count_sentences(text):
    language = detect_language(text)
    if language == 'zh-cn' or language == 'zh-tw':
        # Simple sentence splitting for Chinese
        return len(re.split(r'[，。！？]', text))
    else:
        return len(sent_tokenize(text))
    
def process_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Process based on file type
    if file.content_type == 'application/json' or filename.endswith('.json'):
        try:
            json_content = json.loads(content)
            text_fields = []
            
            def extract_text(obj):
                if isinstance(obj, str):
                    text_fields.append(obj)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_text(item)
                elif isinstance(obj, dict):
                    for value in obj.values():
                        extract_text(value)
            
            extract_text(json_content)
            content = ' '.join(text_fields)
        except json.JSONDecodeError:
            print('Error parsing JSON, treating as plain text')
    elif file.content_type == 'application/xml' or filename.endswith('.xml'):
        try:
            root = ET.fromstring(content)
            print (root)
            
            def remove_tags(element):
                text = element.text or ''
                for sub_element in element:
                    text += remove_tags(sub_element)
                text += element.tail or ''
                return text
            
            content = remove_tags(root)
        except ET.ParseError:
            print('Error parsing XML, treating as plain text')
    else:
        # For all other file types, including plain text
        # We'll use the content as is, including utf-8 characters
        # filter out non-utf-8 characters
        content = content.encode('utf-8', 'ignore').decode('utf-8')
    
    # Process the content
    tokens = tokenize_and_stem(content)
    char_count = len(content)
    word_count = len(tokens)
    sentence_count = count_sentences(content)
    keyword_frequency = {}
    for token in tokens:
        keyword_frequency[token] = keyword_frequency.get(token, 0) + 1
    
    # Add to inverted index
    for position, token in enumerate(tokens):
        if token not in index['inverted_index']:
            index['inverted_index'][token] = {}
        if filename not in index['inverted_index'][token]:
            index['inverted_index'][token][filename] = []
        index['inverted_index'][token][filename].append(position)
    
    # Add to document store
    index['document_store'][filename] = {
        'content': content,
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'keyword_frequency': keyword_frequency,
    }
    
    # Save updated index
    save_index()
    
    return {
        'filename': filename,
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'keyword_frequency': dict(sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
    }

def search(query):
    query_language = detect_language(query)
    query_tokens = tokenize_and_stem(query)

    results = {}

    # Process query tokens
    for token in query_tokens:
        print ('token:', token)
        if token in index['inverted_index']:
            for doc_id in index['inverted_index'][token]:
                if doc_id not in results:
                    results[doc_id] = {'score': 0, 'matches': {}}
                results[doc_id]['score'] += len(index['inverted_index'][token][doc_id])
                results[doc_id]['matches'][token] = len(index['inverted_index'][token][doc_id])

    # Process phrases (for both Chinese and English)
    phrases = re.findall(r'"([^"]*)"', query)
    for phrase in phrases:
        print ('phrase:', phrase)
        phrase_tokens = tokenize_and_stem(phrase)
        for doc_id in index['document_store']:
            doc_content = index['document_store'][doc_id]['content']
            if query_language == 'zh-cn' or query_language == 'zh-tw':
                # For Chinese, use direct string matching
                matches = doc_content.count(phrase)
            else:
                # For English, use regex to match stemmed tokens
                phrase_regex = r'\b' + r'\s+'.join(map(re.escape, phrase_tokens)) + r'\b'
                matches = len(re.findall(phrase_regex, doc_content, re.IGNORECASE))
            if matches > 0:
                if doc_id not in results:
                    results[doc_id] = {'score': 0, 'matches': {}}
                results[doc_id]['score'] += matches * len(phrase_tokens)
                results[doc_id]['matches'][phrase] = matches

    return sorted([
        {
            'filename': doc_id,
            'score': data['score'],
            'matches': data['matches'],
            'char_count': index['document_store'][doc_id]['char_count'],
            'word_count': index['document_store'][doc_id]['word_count'],
            'sentence_count': index['document_store'][doc_id]['sentence_count'],
            'keyword_frequency': dict(sorted(index['document_store'][doc_id]['keyword_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]),
            'preview': index['document_store'][doc_id]['content'][:250] + '...'
        }
        for doc_id, data in results.items()
    ], key=lambda x: x['score'], reverse=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_info = process_file(file)
        return jsonify(file_info)

@app.route('/search')
def search_documents():
    query = request.args.get('q')
    if not query:
        return jsonify({'results': []})
    results = search(query)
    return jsonify({'results': results})

@app.route('/documents/<path:filename>')
def serve_document(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Script started.")
    print(f"Current working directory: {os.getcwd()}")
    # print(f"All environment variables: {dict(os.environ)}")
    
    port_env = os.environ.get('PORT')
    print(f"PORT environment variable: {port_env}")
    
    try:
        port = int(port_env) if port_env is not None else 8080
    except ValueError:
        print(f"Error: PORT environment variable '{port_env}' is not a valid integer. Using default port 8080.")
        port = 8080
    
    print(f"Converted port: {port}")

    if port == 80:
        print("Port is 80, attempting to set up HTTPS...")
        # HTTPS configuration
        
        try:
            cert_path = 'C:\\Certbot\\live\\to-ai.net-0001\\'
            print(f"Certificate path: {cert_path}")
            print(f"Checking if certificate files exist:")
            print(f"  Private key: {os.path.exists(os.path.join(cert_path, 'privkey.pem'))}")
            print(f"  Full chain: {os.path.exists(os.path.join(cert_path, 'fullchain.pem'))}")
            
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(
                certfile=os.path.join(cert_path, 'fullchain.pem'),
                keyfile=os.path.join(cert_path, 'privkey.pem')
            )
            print("SSL context created successfully.")
            # Run HTTPS server
            print("Starting HTTPS server on port 443...")
            app.run(host='0.0.0.0', port=443, ssl_context=context)
        except Exception as e:
            print(f"Error setting up HTTPS: {e}", file=sys.stderr)
            print("SSL Error details:", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print("Falling back to HTTP...", file=sys.stderr)
            app.run(host='0.0.0.0', port=80)
    else:
        # Run HTTP server on the specified port
        print(f"Starting HTTP server on port {port}...")
        app.run(host='0.0.0.0', port=port)