import os
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
import torch
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from transformers import CLIPProcessor, CLIPModel
from chromadb import EmbeddingFunction
import requests
from PIL import Image
import io
import numpy as np

from src.kg_manager import kg_manager

load_dotenv()

class ProfCLIPEmbeddingFunction(EmbeddingFunction):
    """
    Incapsula il modello CLIP di Hugging Face per renderlo compatibile con ChromaDB.
    Gestisce sia il testo (per le query) sia le immagini (per l'inserimento).
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Caricamento modello {model_name} su {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, input) -> list[list[float]]:
        embeddings = []
        
        
        if isinstance(input[0], str):
            inputs = self.processor(text=input, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_text_features(**inputs)
        
        else:
            inputs = self.processor(images=input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
        
        if not isinstance(embeds, torch.Tensor):
            if hasattr(embeds, 'pooler_output') and embeds.pooler_output is not None:
                embeds = embeds.pooler_output
            elif hasattr(embeds, 'image_embeds') and embeds.image_embeds is not None:
                embeds = embeds.image_embeds
            elif hasattr(embeds, 'text_embeds') and embeds.text_embeds is not None:
                embeds = embeds.text_embeds
            else:
                embeds = embeds[0]

        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        
        return embeds.cpu().tolist()

class RAGManager:
    """
    Gestisce il Vector Database (ChromaDB) per il retrieval di documenti turistici.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db_travel"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.active_topic = None
        self._bm25_needs_update = True
        self.clip_embedding_fn = ProfCLIPEmbeddingFunction()
        self.image_collection = None
        self._initialize()
    
    def _initialize(self):
        """Inizializza o carica il Vector DB. Se è il primo avvio, parte vuoto."""
        import os
        
        db_exists = os.path.exists(self.persist_directory)
        
        if not db_exists:
            print("[RAG Manager] Primo avvio: Nessun Vector DB trovato. Creazione archivio vuoto in corso...")
        else:
            print("[RAG Manager] Vector DB trovato sul disco. Caricamento memoria precedente in corso...")

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        try:
            doc_count = len(self.vector_store.get()['ids'])
            print(f"[RAG Manager] Vector DB pronto! Attualmente contiene {doc_count} frammenti di testo.")
        except Exception:
            print("[RAG Manager] Vector DB pronto e vuoto.")
        
        if self.vector_store:
            chroma_client = self.vector_store._client
            self.image_collection = chroma_client.get_or_create_collection(
                name="travel_images", 
                embedding_function=self.clip_embedding_fn
            )
            
            img_count = self.image_collection.count()
            print(f"[RAG Manager] Collection immagini CLIP pronta! Attualmente contiene {img_count} immagini.")
            
            self._sync_bm25()
    
    def _sync_bm25(self):
        """Ricostruisce l'indice BM25 in memoria leggendo i documenti da ChromaDB."""
        if self.vector_store is None:
            return
            
        print("[RAG Manager] Sincronizzazione dell'indice lessicale BM25 in corso (Lightweight)...")
        try:
           
            db_data = self.vector_store.get(include=["documents", "metadatas"])
            
            documents = []
           
            for doc_text, meta in zip(db_data['documents'], db_data['metadatas']):
                doc = Document(
                    page_content=doc_text,
                    metadata=meta if meta else {}
                )
                documents.append(doc)
            
            if documents:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                print(f"[RAG Manager] BM25 sincronizzato con {len(documents)} documenti.")
            
            self._bm25_needs_update = False
            
        except Exception as e:
            print(f"[RAG Manager] Errore durante la sincronizzazione di BM25: {e}")


    def search(self, query: str, k: int = 3) -> list:
        """
        Esegue una ricerca ibrida (Semantica Chroma + Lessicale BM25) 
        con filtro di diversità delle fonti per evitare la "Chunk Domination".
        """
        if self.vector_store is None:
            return []
        print("provaaaaaaaa")
        if self._bm25_needs_update:
            self._sync_bm25()
            

        fetch_k = k * 5 
        
        if self.bm25_retriever is None:
            raw_docs = self.vector_store.similarity_search(query, k=fetch_k)
        else:
            self.bm25_retriever.k = fetch_k
            chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": fetch_k})
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            raw_docs = self.ensemble_retriever.invoke(query)
            
        diverse_docs = []
        seen_sources = {}
        MAX_CHUNKS_PER_SOURCE = 2
        
        for doc in raw_docs:
            source = doc.metadata.get("source", "Sconosciuta")
            
            if seen_sources.get(source, 0) < MAX_CHUNKS_PER_SOURCE:
                diverse_docs.append(doc)
                seen_sources[source] = seen_sources.get(source, 0) + 1
                
            if len(diverse_docs) >= k:
                break
                
        if len(diverse_docs) < k:
            for doc in raw_docs:
                if doc not in diverse_docs:
                    diverse_docs.append(doc)
                if len(diverse_docs) >= k:
                    break
                    
        return diverse_docs
    
    def add_documents(self, documents: list):
        """
        Aggiunge nuovi documenti al Vector DB.
        Utile per integrare articoli scaricati dal web.
        """
        if self.vector_store is None:
            print("[RAG Manager] Vector DB non inizializzato.")
            return
        
        self.vector_store.add_documents(documents)
        print(f"[RAG Manager] Aggiunti {len(documents)} documenti al Vector DB.")
        
        
        self._bm25_needs_update = True 
    
    def format_search_results(self, docs: list) -> str:
        """
        Formatta i risultati di ricerca in una stringa leggibile per l'LLM.
        Gestisce dinamicamente documenti dal Dataset, dal Web Scraper e dal Blog.
        """
        if not docs:
            return "Nessun documento rilevante trovato nell'archivio locale."
        
        response = " Documenti recuperati dall'archivio locale:\n\n"
        for idx, d in enumerate(docs, 1):
            response += f"--- Documento {idx} ---\n"
            
            if "title" in d.metadata:
                response += f"Titolo Pagina Web: {d.metadata.get('title', 'Sconosciuto')}\n"
                response += f"Fonte (URL): {d.metadata.get('source', 'Sconosciuta')}\n"
                
            elif "name" in d.metadata:
                response += f"Attrazione/Topic: {d.metadata.get('name', 'N/A')}\n"
                response += f"Tipo: {d.metadata.get('type', 'N/A')}\n"
                
                if "city" in d.metadata:
                    response += f"Città: {d.metadata.get('city')}\n"
                if "prefecture" in d.metadata:
                    response += f"Prefettura: {d.metadata.get('prefecture')}\n"
            
            response += f"Contenuto: {d.page_content[:1800]}...\n\n"
        
        return response
    
    def add_images_from_urls(self, urls: list, context_query: str):
        """Scarica immagini, scarta le icone e le passa a CLIP per la vettorializzazione."""
        if not self.image_collection: return
        
        valid_images = []
        metadatas = []
        ids = []
        
        for url in urls:
            
            img_id = hashlib.md5(url.encode('utf-8')).hexdigest()
            
            existing = self.image_collection.get(ids=[img_id])
            if existing and existing['ids']:
                continue
                
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=3)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    
                    if img.width > 400 and img.height > 300:
                        valid_images.append(np.array(img))
                        metadatas.append({"url": url, "context": context_query})
                        ids.append(img_id)
            except Exception:
                continue 
                
        if valid_images:
            self.image_collection.upsert(
                images=valid_images,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG Manager] Salvate {len(valid_images)} NUOVE immagini vettorializzate con CLIP.")

    def search_image(self, text_query: str, exclude_urls: list = None) -> str:
        """Il Drafter usa questo metodo: passa del testo, CLIP trova l'immagine associata, scartando i doppioni."""
        if not self.image_collection or self.image_collection.count() == 0:
            return ""
            
        exclude_urls = exclude_urls or []
            
        results = self.image_collection.query(
            query_texts=[text_query],
            n_results=5
        )
        
        if results['metadatas'] and results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                img_url = metadata.get('url')
                if img_url and img_url not in exclude_urls:
                    return img_url
                    
        return ""

rag_manager = RAGManager()