from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import json
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import jieba
import re
from collections import defaultdict, Counter
import ssl
import sys
import string

# 起始app
app = Flask(__name__)
root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
# cors防止跨域
CORS(app, resources={r"/*": 
                     {"origins": "*",
                    "methods": ["GET", "POST"],
                    "allow_headers": ["Content-Type", "Authorization"]}})  # You can replace "*" with specific domains

app.config['UPLOAD_FOLDER'] = os.path.join(root_path, 'public', 'documents')
# create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 下載nltk資料
if root_path == '/home/www':
     # ubuntu server用此路徑
    nltk_path = '/home/nltk_data'
    os.makedirs(nltk_path, exist_ok=True)
    nltk.data.path.append(nltk_path)
    nltk.download('punkt', download_dir=nltk_path)
    nltk.download('punkt_tab', download_dir=nltk_path)
    nltk.download('stopwords', download_dir=nltk_path)
else: 
    # windows和pythonanywhere用此路徑
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Initialize NLTK components
snowball_stemmer =  SnowballStemmer("english", ignore_stopwords=True)
english_stop_words = set(stopwords.words('english'))

# Load Chinese stop words 
with open(os.path.join(root_path, 'chinese_stop_words.txt'), 'r', encoding='utf-8') as f:
    chinese_stop_words = set(f.read().splitlines())

# Initialize Jieba
jieba.set_dictionary(os.path.join(root_path, 'dict.txt.big'))

# 設定 index structure
index = {
    'inverted_index': {},
    'document_store': {}
}

# 設定資料庫檔案路徑
DB_FILE = os.path.join(root_path, 'local_database.json')

# 載入資料庫
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

# 儲存資料庫
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
    # Remove punctuation，保留文字和數字及空格
    wordsCount = 0
    text = re.sub(r'[^\w\s-]', '', text)    
    language = detect_language(text)
    print ('language:', language)
    if language == 'zh-cn' or language == 'zh-tw':
        # Use jieba for Chinese tokenization
        tokens = jieba.cut_for_search(text)
        print ('tokens:', tokens)
        wordsCount = len(tokens)
        return[ [token for token in tokens if token not in chinese_stop_words and token.strip()],
             wordsCount]
    else:
        tokens = word_tokenize(text.lower())
        wordsCount = len(tokens)
        # return tokens
        return [
             [snowball_stemmer.stem(token) for token in tokens if token not in english_stop_words and token.strip()],
             wordsCount
        ]



def count_sentences(text):
    language = detect_language(text)
    if language == 'zh-cn' or language == 'zh-tw':
        # Simple sentence splitting for Chinese
        return len(re.split(r'[，。！？]', text))
    else:
        return len(sent_tokenize(text))

def count_non_ascii_chars(text):
    return sum(1 for char in text if ord(char) > 127)

def count_non_ascii_words(tokens):
    nonascii = 0
    for word in tokens:
        for char in word:
            if ord(char) > 127:
                nonascii += 1
                break
    return nonascii

def process_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # remove >2 spaces to 1 space
        content = re.sub(r'\s{2,}', ' ', content)
    
    # Process based on file type
    if file.content_type == 'application/json' or filename.lower().endswith('.json'):
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
    elif file.content_type == 'application/xml' or filename.lower().endswith('.xml'):
        try:
            root = ET.fromstring(content)

            # Extract title and abstract
            title = ''
            abstract = ''
            # find the title or Title element
            title_elem = root.find('.//ArticleTitle')
            print ('title_elem:', title_elem)
            if title_elem is not None:
                title = title_elem.text.strip()
            # find all AbstractText element
            abstract_elem = root.findall('.//AbstractText')
            abstractStringList = []
            for elem in abstract_elem:  
                abstract = ET.tostring(elem, encoding='utf-8', method='text').decode('utf-8')
                abstract = re.sub(r'<.*?>', ' ', abstract)
                abstractStringList.append(abstract)
            abstract = '\n'.join(abstractStringList)
            content = title + '\n' + abstract
                
        except ET.ParseError:
            print('Error parsing XML, treating as plain text')
    elif file.content_type == 'text/plain' or filename.lower().endswith('.txt'):
        # Plain text file
        content = content.encode('utf-8', 'ignore').decode('utf-8')

    # Process the content
    tokens, word_count = tokenize_and_stem(abstract)
    tokensTitle, _ = tokenize_and_stem(title)
    tokens += tokensTitle

    char_count_with_spaces = len(abstract)
    char_count_without_spaces = len(abstract.replace(" ", ""))
    sentence_count = count_sentences(abstract)
    non_ascii_char_count = count_non_ascii_chars(abstract)
    non_ascii_word_count = count_non_ascii_words(abstract)
    keyword_frequency = Counter(tokens)
    
    # remove inverted index if exists in the index
    for token in index['inverted_index']:
        if filename in index['inverted_index'][token]:
            del index['inverted_index'][token][filename]
            
    # Add to inverted index
    for position, token in enumerate(tokens):
        if token not in index['inverted_index']:
            index['inverted_index'][token] = {}
        if filename not in index['inverted_index'][token]:
            index['inverted_index'][token][filename] = []
        index['inverted_index'][token][filename].append(position)
    
    # Add to document store

    index['document_store'][filename] = {
        'title': title if title else "No title",
        'abstract': abstract if abstract else "No abstract",
        'content': content,
        'char_count_with_spaces': char_count_with_spaces,
        'char_count_without_spaces': char_count_without_spaces,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'non_ascii_char_count': non_ascii_char_count,
        'non_ascii_word_count': non_ascii_word_count,
        'keyword_frequency': dict(keyword_frequency),
    }
    
    # Save updated index
    save_index()
    
    return {
        'filename': filename,
        'char_count_with_spaces': char_count_with_spaces,
        'char_count_without_spaces': char_count_without_spaces,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'non_ascii_char_count': non_ascii_char_count,
        'non_ascii_word_count': non_ascii_word_count,
        'keyword_frequency': dict(keyword_frequency.most_common(10))
    }
    

def search(query):
    
    
    # Parse the query
    and_groups = re.findall(r'\+\((.*?)\)', query)
    or_groups = re.findall(r'\|\((.*?)\)', query)
    not_terms = re.findall(r'-(\w+)', query)
    
    # Remove special syntax from the main query
    main_query = re.sub(r'\+\(.*?\)|\|\(.*?\)|-\w+', '', query).strip()
    print('main_query:', main_query)
    
    results = defaultdict(lambda: {'score': 0, 'matches': {}, 'indexList': []})

    # Process main query
    main_query_tokens = process_query_terms(main_query, results)

    print('main_results:', results)

    # Process AND groups
    for and_group in and_groups:
        and_results = defaultdict(lambda: {'score': 0, 'matches': {}})
        process_query_terms(and_group, and_results)
        print('and_results:', and_results)
        results = {doc_id: data for doc_id, data in results.items() if doc_id in and_results}
        # Update scores and matches
        for doc_id, data in results.items():
            data['score'] += and_results[doc_id]['score']
            data['matches'].update(and_results[doc_id]['matches'])


    # Process OR groups
    for or_group in or_groups:
        or_results = defaultdict(lambda: {'score': 0, 'matches': {}})
        process_query_terms(or_group, or_results)
        print('or_results:', or_results)
        results = {doc_id: data for doc_id, data in results.items()} | {doc_id: data for doc_id, data in or_results.items() if doc_id not in results}
        for doc_id, data in results.items():
            if doc_id in or_results:
                data['score'] += or_results[doc_id]['score']
                data['matches'].update(or_results[doc_id]['matches'])

    # Process NOT terms
    for not_term in not_terms:
        not_results =  defaultdict(lambda: {'score': 0, 'matches': {}})
        print('not_results:', not_results)
        process_query_terms(not_term, not_results)
        results = {doc_id: data for doc_id, data in results.items() if doc_id not in not_results}


    # Convert defaultdict to regular dict for JSON serialization
    results = dict(results)
    print('final_results:', results)


    for doc_id, data in results.items():
        def highlight_tokens(text, tokens):
            position = -1
            for token in tokens:
               
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                
                next = pattern.search(text, position + 1)
                newPosition = next.start() if next else -1
                print ('newPosition:', newPosition)
                while newPosition > position:
                    # find  the first space after newPosition
                    spacePosition = text.find(' ', newPosition)
                    targetString = text[newPosition:spacePosition]
                    text = text[:newPosition] + '<span class="bg-yellow-200">' + targetString + '</span>' + text[spacePosition:]
                    position = newPosition + len(targetString)
                    next = pattern.search(text, position)
                    newPosition = next.start() if next else -1

                    
            
            return text
        title = index['document_store'][doc_id]['title']
        abstract = index['document_store'][doc_id]['abstract']
        # highlight all the match main_query_tokens in the abstract and title with <span class="bg-yellow-200"> tag
        
        data['abstract'] = highlight_tokens(abstract, main_query_tokens)
        data['title'] = highlight_tokens(title, main_query_tokens)

        



    return sorted([
        {
            'filename': doc_id,
            'title': data['title'],
            'abstract': data['abstract'],
            'score': data['score'],
            'matches': data['matches'],
            'char_count_with_spaces': index['document_store'][doc_id]['char_count_with_spaces'],
            'char_count_without_spaces': index['document_store'][doc_id]['char_count_without_spaces'],
            'word_count': index['document_store'][doc_id]['word_count'],
            'sentence_count': index['document_store'][doc_id]['sentence_count'],
            'non_ascii_char_count': index['document_store'][doc_id]['non_ascii_char_count'],
            'non_ascii_word_count': index['document_store'][doc_id]['non_ascii_word_count'],
            'keyword_frequency': dict(sorted(index['document_store'][doc_id]['keyword_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]),
            
        }
        for doc_id, data in results.items()
    ], key=lambda x: x['score'], reverse=True)

def process_query_terms(query, results):
    query_tokens = tokenize_and_stem(query)[0]
    for token in query_tokens:
        if token in index['inverted_index']:
            for doc_id in index['inverted_index'][token]:
                print ('doc_id:', doc_id, 'token:', token, 'score:', len(index['inverted_index'][token][doc_id]))
                results[doc_id]['score'] += len(index['inverted_index'][token][doc_id])
                results[doc_id]['matches'][token] = len(index['inverted_index'][token][doc_id])
    return query_tokens




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    accepted_types = ['json', 'xml', 'txt']
    if file and file.filename.split('.')[-1].lower() in accepted_types:
        file_info = process_file(file)
        return jsonify(file_info)
    else:
        return jsonify({'error': '不支援的檔案格式'}), 400


@app.route('/search', methods=['POST'])
def search_documents():
    data = request.get_json()
    query = data.get('q')
    print('query:', query)
    if not query:
        return jsonify({'error': '請輸入搜尋字串'}), 400
    try:
        results = search(query)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/documents/<path:filename>')
def serve_document(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete', methods=['POST'])
def delete_document():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        # Remove the file
        os.remove(filepath)
        
        # Remove the document from the index
        if filename in index['document_store']:
            del index['document_store'][filename]
        
        # Remove the document from the inverted index
        for token in index['inverted_index']:
            if filename in index['inverted_index'][token]:
                del index['inverted_index'][token][filename]
        
        # Remove empty tokens from the inverted index
        index['inverted_index'] = {token: docs for token, docs in index['inverted_index'].items() if docs}
        
        # Save the updated index
        save_index()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/documents', methods=['POST'])
def get_all_documents():
    documents = []
    for filename, data in index['document_store'].items():
        documents.append({
            'filename': filename,
            'char_count_with_spaces': data['char_count_with_spaces'],
            'char_count_without_spaces': data['char_count_without_spaces'],
            'word_count': data['word_count'],
            'sentence_count': data['sentence_count'],
            'non_ascii_char_count': data['non_ascii_char_count'],
            'non_ascii_word_count': data['non_ascii_word_count']
        })
    return jsonify(documents)

@app.route('/documents')
def document_list():
    return render_template('document_list.html')

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
            # Certificate is saved at: /etc/letsencrypt/live/to-ai.net/fullchain.pem
            # Key is saved at:         /etc/letsencrypt/live/to-ai.net/privkey.pem

            # check if the certificate files exist
            # ubuntu server
            if os.path.exists('/etc/letsencrypt/live/to-ai.net/fullchain.pem') and os.path.exists('/etc/letsencrypt/live/to-ai.net/privkey.pem'):
                cert_path = '/etc/letsencrypt/live/to-ai.net/'
            # windows server
            elif os.path.exists('C:\\Certbot\\live\\to-ai.net-0001\\fullchain.pem') and os.path.exists('C:\\Certbot\\live\\to-ai.net-0001\\privkey.pem'):
                cert_path = 'C:\\Certbot\\live\\to-ai.net-0001\\'
            else:
                raise FileNotFoundError("Certificate files not found in default locations.")

            print(f"Certificate path: {cert_path}")
            
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

