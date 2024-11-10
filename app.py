from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import json
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import jieba
import re
from collections import defaultdict, Counter
import ssl
import sys
import string
import Levenshtein
import io

import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from flask import Response
from word_to_vec_model_pubmed import load_documents, PubmedProcessor, VectorVisualizer, AdvancedVisualizer

# 起始app
app = Flask(__name__, static_folder='javascript')
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB limit
root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)


# Global variables to store processors for different datasets
processors = {
    'caregiver': None,
    'psychosis': None
}




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
    nltk.download('wordnet', download_dir=nltk_path)
else: 
    # windows和pythonanywhere用此路徑
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
english_stop_words = set(stopwords.words('english'))

@app.route('/w2v')
def index():
    return render_template('w2v.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    dataset = data['dataset']
    params = data['params']
    
    def generate():
        try:
            # Load documents based on dataset selection
            folder_path = os.path.join(root_path, 'pubmed_ad_cargiver222324') if dataset == 'caregiver' else os.path.join(root_path, 'pubmed_ad_psychosis')
            yield json.dumps({"status": "loading", "message": "Loading documents..."}) + "\n"
            
            documents = load_documents(folder_path)
            yield json.dumps({"status": "processing", "message": "Processing documents..."}) + "\n"
            
            # Initialize processor
            processor = PubmedProcessor()
            processor.load_documents(documents)
            
            yield json.dumps({"status": "training", "message": "Training Word2Vec model..."}) + "\n"
            
            # Train model with parameters
            processor.train_word2vec(
                vector_size=int(params['vectorSize']),
                window=int(params['windowSize']),
                min_count=int(params['minCount']),
                sg=int(params['architecture'])
            )
            
            # Store processor globally
            processors[dataset] = processor

            # get document count, word count, and vocabulary size
            document_count = processor.model.corpus_count
            word_count = processor.model.corpus_total_words
            vocabulary_size = len(processor.model.wv.key_to_index)

            
            yield json.dumps({"status": "complete", "message": "Training complete!\n文件數量: {}\n總字數: {}\n詞彙數: {}".format(document_count, word_count, vocabulary_size)}) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/similar-words', methods=['POST'])
def find_similar_words():
    data = request.json
    word = data['word']
    dataset = data['dataset']
    topn = int(data.get('topn', 10))
    
    if not processors.get(dataset):
        return jsonify({"error": "Model not trained yet"}), 400
        
    similar_words = processors[dataset].find_similar_words(word, topn)
    return jsonify({"results": similar_words})

@app.route('/api/suggest-words', methods=['POST'])
def suggest_words():
    data = request.json
    context = data['context']
    dataset = data['dataset']
    topn = int(data.get('topn', 5))
    
    if not processors.get(dataset):
        return jsonify({"error": "Model not trained yet"}), 400
        
    suggestions = processors[dataset].suggest_words(context, topn)
    return jsonify({"results": suggestions})

@app.route('/api/similar-documents', methods=['POST'])
def similar_documents():
    data = request.get_json()
    query = data.get('query', '')
    top_n = data.get('top_n', 5)    
    dataset = data['dataset']    
    
    if not processors.get(dataset):
        return jsonify({"error": "Model not trained yet"}), 400
    # 獲取相似文檔
    results = processors[dataset].query_similar_documents(query, top_n=top_n)
    
    # 準備返回結果
    documents = []
    for doc, score in results:
        documents.append({
            'title': doc['title'],
            'abstract': doc['abstract'],
            'score': float(score)
        })
    return jsonify({'documents': documents})

@app.route('/api/plot/<plot_type>', methods=['POST'])
def get_plot(plot_type):
    data = request.json
    dataset = data['dataset']
    
    if not processors.get(dataset):
        return jsonify({"error": "Model not trained yet"}), 400
        
    # Get figure size based on option
    figure_sizes = {
        'small': (10, 8),
        'medium': (15, 12),
        'large': (20, 16)
    }
    figsize = figure_sizes.get(data.get('figure_size', 'medium'), (15, 12))
    
    visualizer = VectorVisualizer(processors[dataset])
    advanced_visualizer = AdvancedVisualizer(processors[dataset])
    
    # try:
        
    # Close any existing plots if they exist
    # plt.close('all')
    
    if plot_type == 'vector-space':
        plt = visualizer.plot_vector_space(
            n_words=int(data.get('n_words', 100)),
            method=data.get('method', 'tsne'),
            figsize=figsize
        )
    elif plot_type == 'word-neighborhood':
        plt = visualizer.plot_word_neighborhood(
            data['word'],
            n_neighbors=int(data.get('n_neighbors', 10)),
            figsize=figsize
        )
    elif plot_type == 'word-clusters':
        plt = advanced_visualizer.plot_word_clusters(
            n_words=int(data.get('n_words', 100)),
            n_clusters=int(data.get('n_clusters', 5)),
            method=data.get('method', 'umap'),
            interactive=False,
            min_freq=int(data.get('min_freq', 5)),
            font_size=data.get('font_size', 12),
        )
    
    # Adjust font size globally
    plt.rcParams.update({'font.size': data.get('font_size', 12)})
    
    # Convert plot to base64 image
    import io
    import base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return jsonify({"image": img_base64})
        
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 400
    
# Load Chinese stop words 
with open(os.path.join(root_path, 'chinese_stop_words.txt'), 'r', encoding='utf-8') as f:
    chinese_stop_words = set(f.read().splitlines())

# Initialize Jieba
jieba.set_dictionary(os.path.join(root_path, 'dict.txt.big'))

indexKeys = ['inverted_index_abstract', 'inverted_index_title',
              'inverted_index_year', 'inverted_index_author', 'document_store']

# 設定 index structure
porterIndex = { }
nonPorterIndex = { }

# 設定資料庫檔案路徑
DB_FILE = os.path.join(root_path, 'local_database.json')
DB_FILE_NON_PORTER = os.path.join(root_path, 'local_database_non_porter.json')

# 載入資料庫
def load_index(indexKeys, porter=True):
    try:
        if porter:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = f.read()
                if not data:
                    return { key: {} for key in indexKeys}
                index = json.loads(data)
                for key in indexKeys:
                    if key not in index:
                        index[key] = {}
                return index
            print('Loaded existing index from file')
        else:
            with open(DB_FILE_NON_PORTER, 'r', encoding='utf-8') as f:
                data = f.read()
                if not data:
                    return { key: {} for key in indexKeys}
                index = json.loads(data)
                for key in indexKeys:
                    if key not in index:
                        index[key] = {}
                return index
            print('Loaded existing index from file')        
    except FileNotFoundError:
        print('No existing index found, starting with an empty index')
        return { key: {} for key in indexKeys}

    except json.JSONDecodeError:
        print('Error decoding JSON, starting with an empty index')
        return { key: {} for key in indexKeys}

# Load existing index on startup
porterIndex = load_index( indexKeys)
nonPorterIndex = load_index( indexKeys, False)

# 儲存資料庫
def save_index():
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(porterIndex, f, ensure_ascii=False)
        print('Index saved to file')
        with open(DB_FILE_NON_PORTER, 'w', encoding='utf-8') as f:
            json.dump(nonPorterIndex, f, ensure_ascii=False)
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
    
    if (len(text) == 0):
        return 'en'
    # If more than 20% of characters are Chinese, consider it Chinese
    if chinese_char_count / len(text) > 0.2:
        if traditional_char_count > simplified_char_count:
            return 'zh-tw'
        else:
            return 'zh-cn'
    else:
        return 'en'


def tokenize_and_stem(text, porter=True):
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
        for token in tokens:
            if '-' in token:
                subtokens = token.split('-')
                for subtoken in subtokens:
                    if subtoken not in tokens:
                        tokens.append(subtoken)
        # return tokens
        if porter:
            return [
                #  [lemmatizer.lemmatize(token) for token in tokens if token not in english_stop_words and token.strip()],
                # poter stemmer
                [stemmer.stem(token) for token in tokens],# if token not in english_stop_words and token.strip()],
                    wordsCount
            ]
        else:
            return [
                [token for token in tokens],# if token not in english_stop_words and token.strip()],
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
        # print ('content:', content)
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

            return [process_content('', content, [], '', filename)]
        except json.JSONDecodeError:
            print('Error parsing JSON, treating as plain text')
    elif file.content_type == 'application/xml' or filename.lower().endswith('.xml'):
        print ('type:xml')
        print ('content:', content[:100])
        # try:
        root = ET.fromstring(content)
        # print ('root lens'+len(root.findall('.//PubmedArticle')))
        # print ('root:', root)
        files = []
        for article in root.findall('.//PubmedArticle'):
            # print ('article:', article)
            pmid = article.find('.//PMID').text
            title = article.find('.//ArticleTitle')
            abstractAll = article.findall('.//Abstract/AbstractText')
            abstract = [ET.tostring(abstract, method='text', encoding='unicode') for abstract in abstractAll]
            abstract = " ".join(abstract)
            
            abstract = re.sub(r'<.*?>', ' ', abstract)
            authors = article.findall('.//Author')
            authorsAll = []
            for author in authors:
                if author.find('LastName') is not None and author.find('ForeName') is not None:
                    name = author.find('LastName').text + ' ' + author.find('ForeName').text
                    if name and name not in authorsAll:
                        authorsAll.append(name)
            # year [Year, Month, Day]
            yearRoot = article.find('.//ArticleDate')
            if yearRoot is not None:
                year = yearRoot.find('Year').text + '-' + yearRoot.find('Month').text + '-' + yearRoot.find('Day').text
            else:
                year = None
            pmid = pmid if pmid is not None else ""
            title_text = title.text if title is not None else ""
            abstract_text = abstract if abstract is not None else ""
            authors = authorsAll if authorsAll else ""
            year = year if year is not None else ""

            print ('pmid:', pmid, 'title:', title_text, 'abstract:', abstract_text, 'authors:', authors, 'year:', year)
            if title_text and abstract_text:
                xmlFilename = '{}.xml'.format(pmid)
                files.append(process_content(title_text, abstract_text, authors, year, xmlFilename))
                # save article to xml file
                with open('{}/{}'.format(app.config['UPLOAD_FOLDER'], xmlFilename), 'w', encoding='utf-8') as f:
                    f.write(ET.tostring(article, encoding='unicode'))
        # delete the original file
        os.remove(filepath)
        print ('files:', len(files))
        return files
                
        # except ET.ParseError:
        #     print('Error parsing XML, treating as plain text')
    elif file.content_type == 'text/plain' or filename.lower().endswith('.txt'):
        # Plain text file
        abstract = content.encode('utf-8', 'ignore').decode('utf-8')
    
        return [process_content('', abstract, [], '', filename)]

    
def process_content(title, abstract, authors, year, filename):
    for key in indexKeys:
        if key not in porterIndex:
            porterIndex[key] = {}
        if key not in nonPorterIndex:
            nonPorterIndex[key] = {}

        if key == 'inverted_index_abstract':
            # Process the content
            tokens, word_count = tokenize_and_stem(abstract)
            char_count_with_spaces = len(abstract)
            char_count_without_spaces = len(abstract.replace(" ", ""))
            sentence_count = count_sentences(abstract)
            non_ascii_char_count = count_non_ascii_chars(abstract)
            non_ascii_word_count = count_non_ascii_words(abstract)
            abstract_keyword_frequency = Counter(tokens)
            # Add to inverted index
            # set the index[key][token][filename] = count of token in the abstract
            for position, token in enumerate(tokens):                
                if token not in porterIndex[key]:
                    porterIndex[key][token] = {}
                if filename not in porterIndex[key][token]:
                    porterIndex[key][token][filename] = 1
                else:
                    porterIndex[key][token][filename] += 1

            # nonPorterIndex
            tokens, _ = tokenize_and_stem(abstract, False)
            abstract_keyword_frequency_non_porter = Counter(tokens)
            for position, token in enumerate(tokens):                
                if token not in nonPorterIndex[key]:
                    nonPorterIndex[key][token] = {}
                if filename not in nonPorterIndex[key][token]:
                    nonPorterIndex[key][token][filename] = 1
                else:                 
                    nonPorterIndex[key][token][filename] += 1
        elif key == 'inverted_index_title': 
            # Process the content
            tokens, _ = tokenize_and_stem(title)
            title_keyword_frequency = Counter(tokens)
            for position, token in enumerate(tokens):
                if token not in porterIndex[key]:
                    porterIndex[key][token] = {}
                if filename not in porterIndex[key][token]:
                    porterIndex[key][token][filename] = 1
                else:
                    porterIndex[key][token][filename] += 1
            # nonPorterIndex
            tokens, _ = tokenize_and_stem(title, False)
            title_keyword_frequency_non_porter = Counter(tokens)
            for position, token in enumerate(tokens):                
                if token not in nonPorterIndex[key]:
                    nonPorterIndex[key][token] = {}
                if filename not in nonPorterIndex[key][token]:
                    nonPorterIndex[key][token][filename] = 1
                else:              
                    nonPorterIndex[key][token][filename] += 1
        elif key == 'inverted_index_year':
            if year not in porterIndex[key]:
                porterIndex[key][year] = []
            if filename not in porterIndex[key][year]:
                porterIndex[key][year].append(filename)
        elif key == 'inverted_index_author':
            for author in authors:
                tokens = word_tokenize(author)
                tokens.append(author)

                for token in tokens:
                    if token not in porterIndex[key]:
                        porterIndex[key][token] = []
                    if filename not in porterIndex[key][token]:
                        porterIndex[key][token].append(filename)
        
    # merge the keyword_frequency of title and abstract
    keyword_frequency = {**title_keyword_frequency, **abstract_keyword_frequency}
    keyword_frequency = Counter(keyword_frequency)

    # merge the keyword_frequency of title and abstract for nonPorterIndex
    keyword_frequency_non_porter = {**title_keyword_frequency_non_porter, **abstract_keyword_frequency_non_porter}
    keyword_frequency_non_porter = Counter(keyword_frequency_non_porter)


    
    porterIndex['document_store'][filename] = {
        'title': title if title else "No title",
        'abstract': abstract if abstract else "No abstract",
        'author': ", ".join(authors) if len(authors) > 0 else "No author",
        'year': year if year else "No year",
        'char_count_with_spaces': char_count_with_spaces,
        'char_count_without_spaces': char_count_without_spaces,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'non_ascii_char_count': non_ascii_char_count,
        'non_ascii_word_count': non_ascii_word_count,
        'keyword_frequency': dict(keyword_frequency),
    }

    nonPorterIndex['document_store'][filename] = {
        'title': title if title else "No title",
        'abstract': abstract if abstract else "No abstract",
        'author': ", ".join(authors) if len(authors) > 0 else "No author",
        'year': year if year else "No year",
        'char_count_with_spaces': char_count_with_spaces,
        'char_count_without_spaces': char_count_without_spaces,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'non_ascii_char_count': non_ascii_char_count,
        'non_ascii_word_count': non_ascii_word_count,
        'keyword_frequency': dict(keyword_frequency_non_porter),
    }   
    
    # Save updated index
    # save_index()

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
    

def search(query, porter=True):
    
    # Parse the query
    and_groups = re.findall(r'\+\((.*?)\)', query)
    or_groups = re.findall(r'\|\((.*?)\)', query)
    not_terms = re.findall(r'-(\w+)', query)
    
    # Remove special syntax from the main query
    main_query = re.sub(r'\+\(.*?\)|\|\(.*?\)|-\w+', '', query).strip()
    # print('main_query:', main_query)
    
    results = defaultdict(lambda: {'score': 0, 'matches': {}})

    # Process main query
    main_query_tokens = process_query_terms(main_query, results, porter=porter)
    
    index = porterIndex if porter else nonPorterIndex

    # If no results, suggest similar words
    if len(results) == 0:
        suggestions = []
        for token in main_query_tokens:
            similar_words = find_similar_words(token, index[indexKeys[0]])            
            print ('similar_words:', similar_words)
            for i in range (min (3, len(similar_words))):
                suggestions.append(similar_words[i][0])
          
        return {
            'results': [],
            'suggestions': suggestions
        }
       


    # print('main_results:', results)

    # Process AND groups
    for and_group in and_groups:
        and_results = defaultdict(lambda: {'score': 0, 'matches': {}})
        
        process_query_terms(and_group, and_results, porter=porter)
        # print('and_results:', and_results)
        results = {doc_id: data for doc_id, data in results.items() if doc_id in and_results}
        # Update scores and matches
        for doc_id, data in results.items():
            data['score'] += and_results[doc_id]['score']
            data['matches'].update(and_results[doc_id]['matches'])


    # Process OR groups
    for or_group in or_groups:
        or_results = defaultdict(lambda: {'score': 0, 'matches': {}})
        process_query_terms(or_group, or_results, porter=porter)
        # print('or_results:', or_results)
        results = {doc_id: data for doc_id, data in results.items()} | {doc_id: data for doc_id, data in or_results.items() if doc_id not in results}
        for doc_id, data in results.items():
            if doc_id in or_results:
                data['score'] += or_results[doc_id]['score']
                data['matches'].update(or_results[doc_id]['matches'])

    # Process NOT terms
    for not_term in not_terms:
        not_results =  defaultdict(lambda: {'score': 0, 'matches': {}})
        # print('not_results:', not_results)
        process_query_terms(not_term, not_results, porter=porter)
        results = {doc_id: data for doc_id, data in results.items() if doc_id not in not_results}



    # Convert defaultdict to regular dict for JSON serialization
    results = dict(results)
    # print('final_results:', results)


    for doc_id, data in results.items():
        def highlight_tokens(text, tokens):
            position = -1
            for token in tokens:
               
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                
                next = pattern.search(text, position + 1)
                newPosition = next.start() if next else -1
                # print ('newPosition:', newPosition)
                while newPosition > position:
                    # find  the first space after newPosition
                    delimiters = ' .,;:!?()[]{}-'
                    spacePositions = [text.find(d, newPosition) for d in delimiters]
                    spacePositions = [pos for pos in spacePositions if pos > newPosition]
                    if len(spacePositions) > 0:
                        spacePosition = min(spacePositions)
                    else:
                        spacePosition = len(text)
                    targetString = text[newPosition:spacePosition]
                    text = text[:newPosition] + '<span class="bg-yellow-200">' + targetString + '</span>' + text[spacePosition:]
                    position = newPosition + len(targetString)+35
                    next = pattern.search(text, position)
                    newPosition = next.start() if next else -1

            return text
        title = index['document_store'][doc_id]['title']
        abstract = index['document_store'][doc_id]['abstract']
        # highlight all the match main_query_tokens in the abstract and title with <span class="bg-yellow-200"> tag
        
        data['abstract'] = highlight_tokens(abstract, main_query_tokens)
        data['title'] = highlight_tokens(title, main_query_tokens)



    return {'results': sorted([
        {
            'filename': doc_id,
            'title': data['title'],
            'abstract': data['abstract'],
            'author': index['document_store'][doc_id]['author'],
            'year': index['document_store'][doc_id]['year'],
            'score': data['score'],
            'matches': data['matches'],
            'char_count_with_spaces': index['document_store'][doc_id]['char_count_with_spaces'],
            'char_count_without_spaces': index['document_store'][doc_id]['char_count_without_spaces'],
            'word_count': index['document_store'][doc_id]['word_count'],
            'sentence_count': index['document_store'][doc_id]['sentence_count'],
            'non_ascii_char_count': index['document_store'][doc_id]['non_ascii_char_count'],
            'non_ascii_word_count': index['document_store'][doc_id]['non_ascii_word_count'],
            'keyword_frequency': dict(Counter(index['document_store'][doc_id]['keyword_frequency']).most_common(10))
        }
        for doc_id, data in results.items()
    ], key=lambda x: x['score'], reverse=True)
    }

def process_query_terms(query, results, indexKey=indexKeys[0], mode='AND', porter=True):
    
    query_tokens = tokenize_and_stem(query, porter)[0]
    
    if mode == 'AND':
        # Initialize a set with blank set
        matching_docs = set()
    
    for token in query_tokens:
        # print('token in:', token in index[indexKey])
        index = porterIndex if porter else nonPorterIndex
        
        if token in index[indexKey]:
            # print('indexKey:', indexKey)
            
            if mode == 'OR':
                for doc_id in index[indexKey][token]:
                    # limit the number of matches to 1000
                    # if len(results) >= 1000:
                    #     break
                    # print('doc_id:', doc_id, 'token:', token, 'score:', len(index[indexKey][token][doc_id]))
                    results[doc_id]['score'] += index[indexKey][token][doc_id]
                    results[doc_id]['matches'][token] = index[indexKey][token][doc_id]
            elif mode == 'AND':
                if len(matching_docs) == 0:
                    matching_docs = set(index[indexKey][token].keys())
                    # print('matching_docs:', matching_docs)
                else:
                    # Intersect the current set of matching docs with docs containing this token
                    matching_docs &= set(index[indexKey][token].keys())
                    # print ('matching_docs:', matching_docs)


    
    if mode == 'AND':
        # limit the number of matches to 1000
        # matching_docs = list(matching_docs)[:1000]
        # Update scores only for documents that match all tokens
        for doc_id in matching_docs:
            for token in query_tokens:
                if token in index[indexKey] and doc_id in index[indexKey][token]:
                    results[doc_id]['score'] += index[indexKey][token][doc_id]
                    results[doc_id]['matches'][token] = index[indexKey][token][doc_id]
    
    return query_tokens

def find_similar_words(query_token, index, max_distance=2):
    similar_words = []
    for word in index.keys():
        distance = Levenshtein.distance(query_token.lower(), word.lower())
        if distance <= max_distance:
            similar_words.append((word, distance))
    return sorted(similar_words, key=lambda x: x[1])


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    print ('upload_files')
    if 'files' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    accepted_types = ['json', 'xml', 'txt']
    results = []
    
    for file in files:
        if file and file.filename.split('.')[-1].lower() in accepted_types:
            files = process_file(file)
            for file_info in files:
                results.append(file_info)
        else:
            return jsonify({'error': f'不支援的檔案格式: {file.filename}'}), 400
    save_index()
    return jsonify(results)

BATCH_SIZE = 100
ACCEPTED_TYPES = ['json', 'xml', 'txt']
@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    print ('upload_batch')
    if 'files' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400
    
    results = []
    
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    
    return jsonify(results)

def process_batch(batch):
    results = []
    for file in batch:
        if file and file.filename.split('.')[-1].lower() in ACCEPTED_TYPES:
            # try:
                file_infos = process_file(file)
                for file_info in file_infos:
                    results.append(file_info)
            # except Exception as e:
            #     results.append({'error': f'處理檔案時發生錯誤: {file.filename}', 'details': str(e)})
        else:
            results.append({'error': f'不支援的檔案格式: {file.filename}'})
    save_index()
    return results


@app.route('/search', methods=['POST'])
def search_documents():
    data = request.get_json()
    query = data.get('q')
    porter = data.get('porter', True)
    print ('porter:', porter)
    print('query:', query)
    if not query:
        return jsonify({'error': '請輸入搜尋字串'}), 400
    # try:
    results = search(query, porter)
    return jsonify( results)
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 400

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
    
    # try:
    # Remove the file
    if os.path.exists(filepath):
        os.remove(filepath)   

    for index in [porterIndex, nonPorterIndex]:   
        porter = index == porterIndex
    
        if filename in index['document_store']:
            document = index['document_store'][filename]
        if not document:
            return jsonify({'success': False, 'error': 'Document not found in index'}), 404
        title = document['title']
        abstract = document['abstract']
        author = document['author']
        year = document['year']
    
        # remove title index
        tokens, _ = tokenize_and_stem(title, porter)
        for token in tokens:
            if token in index['inverted_index_title']:
                if filename in index['inverted_index_title'][token]:
                    del index['inverted_index_title'][token][filename]
                    # Remove empty tokens from the inverted index
                    if index['inverted_index_title'][token] == {}:
                        del index['inverted_index_title'][token]

        # remove abstract index
        tokens, _ = tokenize_and_stem(abstract, porter)
        for token in tokens:
            if token in index['inverted_index_abstract']:
                if filename in index['inverted_index_abstract'][token]:
                    del index['inverted_index_abstract'][token][filename]
                    # Remove empty tokens from the inverted index
                    if index['inverted_index_abstract'][token] == {}:
                        del index['inverted_index_abstract'][token]

        # remove author index
        authors = author.split(',')
        authors = [author.strip() for author in authors]
        for author in authors:
            tokens = word_tokenize(author)
            tokens.append(author)
            for token in tokens:
                if token in index['inverted_index_author']:
                    index['inverted_index_author'][token].remove(filename)
                    if index['inverted_index_author'][token] == []:
                        del index['inverted_index_author'][token]

        # remove year index
        if year in index['inverted_index_year']:
            index['inverted_index_year'][year].remove(filename)
            if index['inverted_index_year'][year] == []:
                del index['inverted_index_year'][year]
        

        # Remove the document from the index
        if filename in index['document_store']:
            del index['document_store'][filename]
    
    # Save the updated index
    save_index()
    
    return jsonify({'success': True})
    # except Exception as e:
    #     return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/documents', methods=['POST'])
def get_all_documents():
    documents = []
    for filename, data in porterIndex['document_store'].items():
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
    # print("Script started.")
    # print(f"Current working directory: {os.getcwd()}")
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

