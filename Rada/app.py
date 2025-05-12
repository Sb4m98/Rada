from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
import os
import fitz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from typing import List, Tuple, Dict, Any
import uuid
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from vectorstore_manager import VectorStoreManager
from flask import request, jsonify
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
import json
#from policy_explainer import PolicyExplainer
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
from local_personalizer import LocalPersonalizer
import shutil

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(
    __name__,
    static_folder='static',     
    template_folder='templates'  
)
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DIR = os.path.join(BASE_DIR, 'user')          
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
HIGHLIGHTS_FOLDER = os.path.join(BASE_DIR, 'highlights')
CLU_ENDPOINT = os.getenv("CLU_ENDPOINT")
CLU_KEY = os.getenv("CLU_KEY")
CLU_PROJECT = os.getenv("CLU_PROJECT")
CLU_DEPLOYMENT = os.getenv("CLU_DEPLOYMENT")

for folder in (UPLOAD_FOLDER, HIGHLIGHTS_FOLDER):
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Session storage
sessions = {}
personalizer = LocalPersonalizer()

class DocumentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('italian') + stopwords.words('english'))
    
    def analyze_text(self, text: str) -> Dict:
        # Tokenizzazione e pulizia
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Analisi
        word_count = len(words)
        unique_words = len(set(words))
        word_freq = Counter(words).most_common(10)
        
        # Calcolo della complessità del testo (lunghezza media delle parole)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'word_freq': word_freq,
            'avg_word_length': avg_word_length,
            'vocabulary_density': unique_words / word_count if word_count > 0 else 0
        }
    
class PDFHighlighter:
    def __init__(self):
        self.highlight_color = (1, 0.85, 0)  # Yellow highlight

    def highlight_pdf(self, pdf_path: str, source_docs: List[Any]) -> str:
        """Create a new PDF with highlighted sections."""
        doc = fitz.open(pdf_path)
        
        for source_doc in source_docs:
            # Ensure page number is valid and zero-based
            page_num = source_doc.metadata.get('page', 1)
            if isinstance(page_num, str):
                try:
                    page_num = int(page_num)
                except ValueError:
                    page_num = 1
            
            # Convert to zero-based index and validate
            page_idx = page_num - 1 if page_num > 0 else 0
            if page_idx >= doc.page_count:
                page_idx = doc.page_count - 1
            
            chunk_text = source_doc.page_content
            
            page = doc[page_idx]
            # Get all instances of the text in the page
            text_instances = self.find_text_on_page(page, chunk_text)
            
            # Highlight each instance
            for rect_list in text_instances:
                quads = [rect.quad for rect in rect_list if hasattr(rect, 'quad')]
                
                if quads:  # Only add highlight if we found matches
                    annot = page.add_highlight_annot(quads)
                    annot.set_colors(stroke=self.highlight_color)
                    annot.update()
        
        # Save highlighted PDF
        output_path = os.path.join(HIGHLIGHTS_FOLDER, f"{os.path.basename(pdf_path)}_highlighted.pdf")
        doc.save(output_path)
        doc.close()
        return output_path

    def find_text_on_page(self, page: fitz.Page, search_text: str) -> List[List[fitz.Rect]]:
        """Find all instances of text on the page and return their rectangles."""
        # Clean and prepare the search text
        search_text = self._prepare_text_for_search(search_text)
        
        # Split into sentences for more accurate matching
        sentences = nltk.sent_tokenize(search_text)
        all_matches = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 4:  # Skip very short sentences
                continue
                
            # Search for text instances on the page
            matches = page.search_for(sentence.strip())
            if matches:
                all_matches.append(matches)
        
        return all_matches

    def _prepare_text_for_search(self, text: str) -> str:
        """Clean and prepare text for searching."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove special characters that might interfere with search
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,        # chunk più ampi
            chunk_overlap=500,       # overlap maggiore
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def extract_text_from_pdf(self, pdf_path: str, pdf_index: int) -> List[Tuple[str, Dict[str, Any]]]:
        texts_with_metadata = []
        
        try:
            doc = fitz.open(pdf_path)
            pdf_info = {
                'filename': os.path.basename(pdf_path),
                'pdf_index': pdf_index,
                'total_pages': doc.page_count
            }
            
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                text = page.get_text("text", sort=True).strip()
                if text:
                    text = self._preprocess_text(text)
                    metadata = {
                        'page': page_number + 1,  # Store as 1-based page numbers
                        'pdf_info': pdf_info,
                        'word_locations': page.get_text("words")
                    }
                    texts_with_metadata.append((text, metadata))
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
        
        return texts_with_metadata

    def _preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        return text
    
    def create_chunks(self, texts_with_metadata: List[Tuple[str, Dict]], session_id: str) -> List[Tuple[str, Dict]]:
        chunks_with_metadata = []
        for text, metadata in texts_with_metadata:
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"chunk_{metadata['pdf_info']['pdf_index']}_{metadata['page']}_{i+1}",
                    'text_length': len(chunk),
                    'session_id': session_id  # <-- Aggiungi qui
                })
                chunks_with_metadata.append((chunk, chunk_metadata))
        return chunks_with_metadata
    
    def process_and_analyze(self, pdf_path: str, pdf_index: int) -> Tuple[List[Tuple[str, Dict]], Dict]:
        texts_with_metadata = self.extract_text_from_pdf(pdf_path, pdf_index)
        
        # Analisi del documento
        full_text = ' '.join([text for text, _ in texts_with_metadata])
        analyzer = DocumentAnalyzer()
        analysis_results = analyzer.analyze_text(full_text)
        
        return texts_with_metadata, analysis_results

class RelevanceScorer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calculate the semantic similarity between two texts."""
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

class EnhancedConversationalRetrievalChain(Chain):
    """VImproved version of the conversational retrieval chain."""
    
    retriever: Any
    llm_chain: Any
    question_generator: Any
    memory: Any
    scorer: RelevanceScorer
    min_similarity_threshold: float = 0.75
    
    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer", "source_documents", "relevance_scores"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history = self.memory.chat_memory.messages if self.memory else []
        
        # Generate question considering context
        if chat_history:
            chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
            generated_question = self.question_generator.run(
                question=question,
                chat_history=chat_history_str
            )
        else:
            generated_question = question
        
        # Retrieve documents and calculate relevance
        docs = self.retriever.get_relevant_documents(generated_question)
        
        # Prepare context combining documents
        context = "\n\n".join([d.page_content for d in docs])
        
        # Get response using LLMChain
        response = self.llm_chain.run(
            question=question,
            chat_history=chat_history_str if chat_history else "",
            context=context
        )
        
        # Calculate and validate relevance scores
        relevance_scores = {}
        filtered_docs = []
        
        for doc in docs:
            # Calculate similarity with both question and answer
            question_similarity = self.scorer.compute_similarity(question, doc.page_content)
            answer_similarity = self.scorer.compute_similarity(response, doc.page_content)
            
            # Use weighted average with more weight on answer similarity
            combined_score = (0.3 * question_similarity + 0.7 * answer_similarity)
            
            # Update metadata with validated page information
            if 'page' in doc.metadata:
                # Ensure page numbers are valid
                pdf_info = doc.metadata.get('pdf_info', {})
                total_pages = pdf_info.get('total_pages', 1)
                current_page = doc.metadata['page']
                
                # Adjust page number if it's out of bounds
                if current_page < 1:
                    current_page = 1
                elif current_page > total_pages:
                    current_page = total_pages
                
                doc.metadata['page'] = current_page
            
            doc.metadata['relevance_score'] = combined_score
            relevance_scores[doc.metadata.get('chunk_id', f'chunk_{len(relevance_scores)}')] = {
                'combined_score': combined_score,
                'question_similarity': question_similarity,
                'answer_similarity': answer_similarity,
                'page': doc.metadata.get('page', 1)
            }
            
            if combined_score >= self.min_similarity_threshold:
                filtered_docs.append(doc)
        
        # Sort by combined relevance score
        filtered_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
        
        # Take only the top most relevant documents
        top_docs = filtered_docs[:3]
        
        return {
            "answer": response,
            "source_documents": top_docs,
            "relevance_scores": relevance_scores
        }

    @classmethod
    def from_llm(
        cls,
        llm: Any,
        retriever: Any,
        memory: Any = None,
        **kwargs: Any
    ) -> "EnhancedConversationalRetrievalChain":
        """Builds a new instance of the improved chain."""
        # Create question generator
        question_prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be self-contained.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            
        Independent question:"""
        )
        question_generator = LLMChain(llm=llm, prompt=question_prompt)
        
        # Create main response prompt
        response_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Use the following context to answer the question.
                If you are unsure of the answer, be honest.
                Be specific and cite relevant parts of the context when possible.
            
            Context: {context}
            
            Chat History: {chat_history}
            Request: {question}
            
            Detailed response:"""
        )
        
        # Create the main LLM chain
        llm_chain = LLMChain(llm=llm, prompt=response_prompt)
        
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            question_generator=question_generator,
            memory=memory,
            scorer=RelevanceScorer(),
        )

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'vectorstore': vs_manager.create_new_session_store(session_id),
            'analyses': {},
            'pdf_files': {},
            'conversation': None,
            'chat_history': [
                AIMessage(content="Hi! I'm here to help you analyze your documents. How can I help you?")
            ]
        }
        return session_id
    
    def get_session(self, session_id):
        return self.sessions.get(session_id)


session_manager = SessionManager()
persist_directory="db/chroma"
vs_manager = VectorStoreManager(persist_directory)


# API endpoints remain unchanged
@app.route('/api/create_session', methods=['POST'])
def create_session():
    session_id = session_manager.create_session()
    return jsonify({'session_id': session_id})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 1) Controlli preliminari
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Invalid session ID'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file type'}), 400

    # 2) Salvataggio fisico del PDF
    filename = secure_filename(file.filename)
    pdf_index = len(session['pdf_files'])
    stored_name = f"{session_id}_{filename}"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(pdf_path)

    # 3) Estrazione testo, analisi e chunking
    processor = DocumentProcessor()
    texts_with_metadata, analysis_results = processor.process_and_analyze(pdf_path, pdf_index)
    chunks = processor.create_chunks(texts_with_metadata, session_id)
    
    # 4) Aggiunta dei chunk a ChromaDB e persistenza
    vectorstore = vs_manager.add_chunks(chunks)
    session['vectorstore'] = vectorstore

    # 5) Aggiorna lo stato della sessione (solo metadata)
    session['pdf_files'][pdf_index] = {'path': pdf_path, 'name': filename}
    session['analyses'][filename] = analysis_results

    # 6) Risposta al frontend
    return jsonify({
        'success': True,
        'filename': filename,
        'analysis': analysis_results
    }), 200

@app.route('/api/user/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    # Costruisci il percorso assoluto al file user1.json o user2.json
    filename = f"{user_id}.json"
    filepath = os.path.join(USER_DIR, filename)

    app.logger.debug(f"[Profile] Looking for user profile at: {filepath}")
    if not os.path.isfile(filepath):
        return jsonify({"error": "User not found"}), 404

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        return jsonify(profile)
    except Exception as e:
        app.logger.error(f"[Profile] Error reading profile {filepath}: {e}")
        return jsonify({"error": f"Error reading profile: {str(e)}"}), 500

@app.route('/api/files', methods=['GET'])
def get_files():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
    
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    files = []
    for pdf_index, pdf_info in session['pdf_files'].items():
        files.append({
            'index': pdf_index,
            'name': pdf_info['name'],
            'analysis': session['analyses'].get(pdf_info['name'], {})
        })
    
    return jsonify({'files': files})




@app.route('/api/assist', methods=['POST'])
def insurance_assistant():
    data = request.json or {}
    session_id = data.get("session_id")
    query      = data.get("query", "").strip()
    profile    = data.get("profile", {})

    # Validazioni base
    if not session_id or not query or not profile.get("user_id"):
        return jsonify({"error": "Missing session_id, query or profile"}), 400

    session = session_manager.get_session(session_id)
    vectorstore = session.get("vectorstore")

    # 1) CLU Analysis
    try:
        client = ConversationAnalysisClient(CLU_ENDPOINT, AzureKeyCredential(CLU_KEY))
        with client:
            clu = client.analyze_conversation(task={
                "kind": "Conversation",
                "analysisInput": {"conversationItem": {
                    "participantId": profile["user_id"],
                    "id": "1",
                    "modality": "text",
                    "language": "en-us",
                    "text": query
                }},
                "parameters": {
                    "projectName": CLU_PROJECT,
                    "deploymentName": CLU_DEPLOYMENT,
                    "stringIndexType": "TextElement_V8",
                    "verbose": True
                }
            })
        pred = clu["result"]["prediction"]
        top_intent = pred.get("topIntent", "Unknown")
        entities   = pred.get("entities", [])
        intents_list = pred.get("intents", [])
            # Trova la confidence del top_intent
        intent_score = next(
            (i["confidenceScore"] for i in intents_list if i["category"] == top_intent),
            0.0
        )
    except Exception as e:
        return jsonify({"error": f"CLU Error: {e}"}), 500

    print(clu)
    # 2) Costruisci contesto da documenti (se caricati)
    context = ""
    source_docs = []
    if vectorstore:
        retr = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
        docs = retr.get_relevant_documents(query,
    filter={"session_id": session_id}  # <-- Filtra per sessione
)   
        context = "\n\n".join(d.page_content for d in docs)

        raw_sources = []
        for d in docs:
            md = d.metadata or {}
            stored = md.get("filename") or ""
            # rimuovi il prefisso "<session_id>_" lasciando solo il nome originale
            filename = stored.split("_", 1)[1] if "_" in stored else stored
            raw_sources.append({
                "filename": filename,
                "page":     md.get("page", 1),
                "content":  d.page_content
            })

        # deduplica filename+page
        seen = set()
        for src in raw_sources:
            key = (src["filename"], src["page"])
            if key not in seen:
                seen.add(key)
                source_docs.append(src)
    # 3) Estrai i campi dal profilo
    age             = profile.get("age", "N/A")
    product_info    = profile.get("product", {})
    product_name    = product_info.get("name", "N/A")
    risk_profile    = profile.get("experience", {}).get("risk_profile", "N/A")
    preferred_style = profile.get("preferred_style", "formale")
    app_usage       = profile.get("engagement", {}).get("app_usage_freq", "N/A")
    prefers_chat    = profile.get("engagement", {}).get("prefers_chat", False)

    # Costruisci snippet profilo
    profile_snippet = f"""User profile:
- Age: {age}
- Product: {product_name}
- Risk profile: {risk_profile}
- Preferred style: {preferred_style}
- App usage: {app_usage}
- Prefers chat: {"Yes" if prefers_chat else "No"}"""

    # 4) Prompt LLM includendo profilo, CLU, contesto
    style_instructions = {
        "formal": "Use a formal tone suitable for professional communication. Without use explicitly the name of the user and use sign at the end as Customer Support",
        "colloquial": "Use an informal and friendly tone.",
        "bullet": "Use bullet points with clear and concise phrases."
    }
    ent_text = ", ".join(f"{e['category']} ('{e['text']}')" for e in entities) or "None"

    #llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
    llm = AzureChatOpenAI(azure_deployment="gpt-4o", api_version ="2024-12-01-preview" )
        
    prompt = f"""
You are an insurance assistant.
{profile_snippet}

Detected intent: {top_intent}
Entities: {ent_text}
Context from documents: {context or 'N/A'}

Question: "{query}"

Answer using this style: {style_instructions[preferred_style]}
"""

    answer = llm.invoke(prompt).content
    intents_list = pred.get("intents", [])
    top3 = sorted(intents_list, key=lambda i: i["confidenceScore"], reverse=True)[:3]

    return jsonify({
      "answer": answer,
      "intent": top_intent,
      "intentScore": intent_score,
      "topIntents": top3,          # <-- aggiunto
      "entities": entities,
      "sources": source_docs
    })

@app.route('/api/personalizer/reward', methods=['POST'])
def personalizer_reward():
    data = request.json or {}
    event_id = data.get("eventId")
    reward = data.get("reward")

    if not event_id or reward is None:
        return jsonify({"error": "Missing eventId or reward"}), 400

    try:
        personalizer.send_reward(event_id, float(reward))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/pdf/<path:filename>', methods=['GET'])
def get_pdf(filename):
    return send_file(os.path.join(HIGHLIGHTS_FOLDER, filename))

@app.route('/api/original_pdf/<path:session_id>/<path:filename>', methods=['GET'])
def get_original_pdf(session_id, filename):
    safe_filename = secure_filename(filename)
    return send_file(os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_filename}"))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    session_id = data.get('session_id')
    query = data.get('query')

    # 1) Validazioni
    if not session_id or not query:
        return jsonify({'error': 'Missing session ID or query'}), 400
    session = session_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Invalid session ID'}), 400

    # 2) Configura il retriever di ChromaDB
    retriever = vs_manager.get_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 14, "lambda_mult": 0.7}
    )

    # 3) Inizializza o riusa la catena di conversazione
    if session.get('conversation') is None:
        llm = ChatOpenAI(temperature=0.8, model="gpt-4-turbo-preview")
        #llm = AzureChatOpenAI(azure_deployment="gpt-4o", api_version ="2024-12-01-preview" )
        #llm = ChatOpenAI(    temperature=0.8, model="gpt-4-turbo-preview",model_kwargs={'tools': [your_tools_list], 'tool_choice': 'auto'    } )    # <-- Bisogna inserire la tool_list

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            input_key='question'
        )
        memory.chat_memory.messages = session['chat_history']

        session['conversation'] = EnhancedConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
    # 4) Aggiungi query utente alla cronologia
    session['chat_history'].append(HumanMessage(content=query))

    # 5) Esegui la chain
    result = session['conversation']({"question": query})
    answer = result["answer"]
    source_docs = result["source_documents"]

    # 6) Aggiungi la risposta AI alla cronologia
    session['chat_history'].append(AIMessage(content=answer))

    # 7) Prepara il payload delle fonti evidenziate
    sources_payload = []
    highlighter = PDFHighlighter()
    for doc in source_docs:
        # Usa i metadati "appiattiti"
        pdf_index = doc.metadata.get('pdf_index')
        if pdf_index is None:
            continue

        pdf_info = session['pdf_files'].get(pdf_index)
        if not pdf_info:
            continue

        # evidenzia
        highlighted_path = highlighter.highlight_pdf(pdf_info['path'], [doc])
        highlighted_name = os.path.basename(highlighted_path)

        sources_payload.append({
            'pdf_index': pdf_index,
            'filename': pdf_info['name'],
            'page': doc.metadata.get('page', 1),
            'content': doc.page_content,
            'relevance_score': doc.metadata.get('relevance_score', 0),
            'highlighted_pdf': highlighted_name
        })

    # 8) Restituisci al frontend
    return jsonify({
        'answer': answer,
        'source_documents': sources_payload
    }), 200

@app.route('/api/extract_process', methods=['POST'])
def extract_process():
    data       = request.json or {}
    session_id = data.get('session_id')
    query      = data.get('query','')

    if not session_id or not query:
        return jsonify({"error":"Missing session_id or query"}),400

    session = session_manager.get_session(session_id)
    if not session or session.get('vectorstore') is None:
        return jsonify({"error":"Please, upload documents"}),400

    # Usa il vectorstore specifico della sessione
    retriever = session['vectorstore'].as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    print(query)
    
    context = "\n\n".join([d.page_content for d in docs])
    print(context)
    process_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=
    """
    Return a JSON with the following structure based on the user's request {query} and the context extracted from the document {context}
    (NOTE: include only new or updated information and "Make sure all phases mentioned in the relationships are also included in the list of phases."):

    {{
        "fasi": [
            {{
                "nome": "phase name",
                "node_type": "beginEnd/mainProcess/decision/criticalNode",
                "descrizione": "detailed description",
                "tempistiche": "timeline or deadline if specified",
                "prerequisiti": "necessary conditions if specified",
                "effetti": "consequences of the phase if specified",
                "sottostati": [
                    {{
                        "nome": "substate name",
                        "descrizione": "substate description",
                        "tempistiche": "timeline or deadline if specified",
                        "prerequisiti": "necessary conditions if specified",
                        "effetti": "consequences of the substate if specified"
                    }}
                ]
            }}
        ],
        "relazioni": [
            {{
                "da": "starting phase",
                "a": "target phase",
                "condizione": "condition for the transition if specified"
            }}
        ]
    }}
        The node types must be classified as:
            - beginEnd: for initial/final nodes (e.g., start/end of process)
            - mainProcess: for main process phases
            - decision: for decision points (must have at least two outgoing relationships)
            - criticalNode: for critical nodes with significant impact

        IMPORTANT: Do not include triple backticks or the word "json" in the response. Provide ONLY the valid JSON object.
        Rules:     
        1. Use ONLY information from the context provided
        2. Format the JSON perfectly without errors
        3. The "phases" field MUST be present even if empty
        """

    )


    # 2) Chiamata al LLM
    llm_chain = LLMChain(
       llm=ChatOpenAI(temperature=0, model="gpt-4-turbo-preview"),
       prompt=process_prompt
    )
    result = llm_chain.run(context=context, query=query)
    
    # Rimuovere eventuali code block markers che potrebbero essere nella risposta
    # Questo rimuove ```json, ``` o qualsiasi combinazione di backtick all'inizio o alla fine
    result = re.sub(r'^```json\s*|\s*```$', '', result.strip())
    result = re.sub(r'^```\s*|\s*```$', '', result.strip())

    try:
        process_json = json.loads(result)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic JSON: {result}")
        process_json = {"raw_response": result}

    return jsonify(process_json)

@app.route('/api/reset', methods=['POST'])
def reset_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
    
    # Create a new session with the same ID
    session_manager.sessions[session_id] = {
        'vectorstore': None,
        'analyses': {},
        'pdf_files': {},
        'conversation': None,
        'chat_history': [
            AIMessage(content="Hi! I'm here to help you analyze your documents. How can I help you?")
        ]
    }
    
    # Clean up files for this session
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        if file.startswith(f"{session_id}_"):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    
    # Clean up highlights for this session
    for file in os.listdir(HIGHLIGHTS_FOLDER):
        if file.startswith(f"{session_id}_"):
            os.remove(os.path.join(HIGHLIGHTS_FOLDER, file))

    session_path = os.path.join(vs_manager.persist_directory, session_id)
    if os.path.exists(session_path):
        shutil.rmtree(session_path)
    return jsonify({'success': True})

# Serve Single Page Application
@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
def serve_spa(path):
    # Static file
    if path != '' and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # Otherwise serve index.html
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
