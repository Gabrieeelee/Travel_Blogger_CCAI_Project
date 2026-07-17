import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from src.schemas import TravelRouteInput
from src.rag_manager import rag_manager
from datasets import load_dataset
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.kg_manager import kg_manager
import requests
import math
import torch
import json
import os
import trafilatura
from urllib.parse import urlparse

load_dotenv()



KRAG_GROUNDING_DELIM = "\n===GROUNDING_DOCS===\n"

_processed_urls = set()

_banned_domains = [
    "booking.com", "agoda.com", "tripadvisor", "expedia.com", 
    "airbnb.com", "trivago.com", "hotels.com", "skyscanner", 
    "kayak.com", "getyourguide.com", "klook.com", "viator.com", "japanican.com",
    "trip.com", "insiemeintour.it", "travel365.it", "turismo-giappone.it",
    "tokyotohiroshima.com", "civitatis.com", "tourscanner.com" 
]

def is_valid_content(text: str) -> bool:
    """Verifica che il contenuto non sia corrotto o vuoto."""
    if not text or len(text) < 50:
        return False
    special_chars = sum(1 for c in text if c in '#$%&*+')
    if special_chars / len(text) > 0.3:  
        return False
    return True

@tool
def advanced_web_research(query: str) -> str:
    """
    Esegue una ricerca web profonda mirata al travel. Trova blog di viaggio, 
    recensioni e guide, scarica i testi filtrando i siti commerciali di prenotazione (OTA), 
    li divide in blocchi e li salva nel database vettoriale locale (ChromaDB).
    Inoltre, estrae le immagini pertinenti e le passa a CLIP per l'impaginazione automatica.
    """
    print(f"[Scraper Tool] Avvio ricerca web travel per: '{query}'")
    
    TAVILY_SCORE_THRESHOLD = 0.35   
    tavily = TavilySearchResults(max_results=4)
    results = tavily.invoke({"query": query})
    results_list = results.get("results", []) if isinstance(results, dict) else results
    
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_new_documents = []
    saved_urls = []
    
    def is_informative_travel_blog(text: str) -> bool:
        testo_lower = text.lower()
        if len(text) < 400: 
            return False
            
        spam_words = [
            "aggiungi al carrello", "seleziona le date", "cerca voli", "tour privato", "guadagnare commissioni","cancellazione gratuita" 
        ]
        
        if sum(1 for word in spam_words if word in testo_lower) > 5: 
            return False
            
        return True
    
    for res in results_list:
        url = res.get("url", "")
        title = res.get("title", "Senza Titolo")
        
        tavily_score = res.get("score", 0.0)
        if tavily_score < TAVILY_SCORE_THRESHOLD:
            print(f"[Scraper Tool] Scartato (rilevanza Tavily troppo bassa: {tavily_score:.2f} < {TAVILY_SCORE_THRESHOLD}): {url}")
            _processed_urls.add(url)
            continue
        if not url or url in _processed_urls:
            if url in _processed_urls:
                print(f"[Scraper Tool] Saltato (già processato in questa sessione): {url}")
            continue
            
       
        domain = urlparse(url).netloc.lower()
        if any(banned in domain for banned in _banned_domains):
            print(f"[Scraper Tool] URL scartato (Dominio in blacklist commerciale): {domain}")
            _processed_urls.add(url)
            continue
        
        if url.lower().endswith(".pdf"):
            print(f"[Scraper Tool] URL scartato (Formato PDF non supportato): {url}")
            _processed_urls.add(url)
            continue
        
        print(f"[Scraper Tool] Tentativo di download guida/blog: {title}")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            if response.encoding is None or response.encoding.lower() == 'iso-8859-1':
                response.encoding = response.apparent_encoding
                
            html_text = response.text
            html_bytes = response.content
            
            
            testo_pulito = trafilatura.extract(html_text, include_comments=False, include_tables=True, no_fallback=False)
            if not testo_pulito:
                
                testo_pulito = trafilatura.extract(html_bytes, include_comments=False, include_tables=True, no_fallback=False)
            
            
            soup = BeautifulSoup(html_bytes, "html.parser", from_encoding=response.encoding)
            
            if not testo_pulito:
                testo_pulito = soup.get_text(separator=' ', strip=True)
            
            if not is_informative_travel_blog(testo_pulito):
                print(f"[Scraper Tool] Contenuto scartato (troppo spam o troppo breve): {url}")
                _processed_urls.add(url)
                continue
            
            if not is_valid_content(testo_pulito):
                print(f"[Scraper Tool] Contenuto corrotto: {url}")
                _processed_urls.add(url)
                continue

            image_urls = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src.startswith('http') and not src.endswith('.svg') and not src.endswith('.gif'):
                    image_urls.append(src)
            
            if image_urls:
                print(f"[Scraper Tool] Trovate {len(image_urls)} immagini, invio al modello CLIP...")
                rag_manager.add_images_from_urls(image_urls[:5], context_query=query)
                

            chunks = splitter.split_text(testo_pulito)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=f"META TURISTICA/QUERY: {query}\nTESTO: {chunk}",
                    metadata={
                        "source": url,
                        "title": title,
                        "chunk_id": i
                    }
                )
                all_new_documents.append(doc)
                
            saved_urls.append(title)
            _processed_urls.add(url)
            print(f"[Scraper Tool] Salvati {len(chunks)} chunk testuali puliti da: {url}")
            
        except Exception as e:
            print(f"[Scraper Tool] Errore scraping su {url}: {str(e)}")
            _processed_urls.add(url)
            continue
            
    # SALVATAGGIO
    if all_new_documents:
        MAX_TOTAL_CHUNKS = 1500 
        
        if len(all_new_documents) > MAX_TOTAL_CHUNKS:
            print(f"[Scraper Tool] Attenzione: documento enorme ({len(all_new_documents)} chunk). Troncamento a {MAX_TOTAL_CHUNKS}.")
            all_new_documents = all_new_documents[:MAX_TOTAL_CHUNKS]

        MAX_BATCH_SIZE = 5000
        
        for i in range(0, len(all_new_documents), MAX_BATCH_SIZE):
            batch = all_new_documents[i : i + MAX_BATCH_SIZE]
            rag_manager.add_documents(batch)
            print(f"[Scraper Tool] Salvato batch di {len(batch)} documenti puliti nel RAG.")
            
        search_results = rag_manager.search(query, k=8)
        formatted_search_results = rag_manager.format_search_results(search_results)
        
        return (
            f"Ricerca Web completata. Ho scaricato guide da {len(saved_urls)} fonti nel RAG.\n"
            f"Ecco i dati estratti direttamente dalle nuove fonti appena scaricate:\n\n"
            f"{formatted_search_results}"
        )
    
    return "La ricerca web non ha prodotto blog di viaggio leggibili o sono stati tutti bloccati dai filtri OTA."

@tool
def kg_query_tool(query_string: str) -> str:
    """
    Interroga il Knowledge Graph (Neo4j) e restituisce i COLLEGAMENTI GENERALI
    del grafo per l'entita' richiesta, riconoscendone automaticamente il tipo.
    Supporta piu' entita' separate da virgola ("Tokyo, Kinkaku-ji, Gion Matsuri").

    - City      -> prefettura, regione, spot collegati, festival (citta' + prefettura)
    - Spot      -> citta', prefettura, regione, categorie, spot vicini (NEAR), festival della zona
    - Festival  -> citta', prefettura, regione, spot della zona
    - Prefecture / Region -> collegamenti gerarchici e festival

    Se l'entita' non e' nel grafo lo segnala, cosi' puoi usare i tool Web e RAG.
    """
    entities = [e.strip() for e in query_string.split(",") if e.strip()]
    if not entities:
        return json.dumps({
            "status": "error",
            "message": "Nessuna entita' valida fornita per la ricerca nel KG."
        }, ensure_ascii=False)

    print(f"[Esecuzione Tool] Interrogazione KG per: {entities}")

    def _dedup(seq):
        seen, out = set(), []
        for x in seq or []:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    results = {}
    not_found = []
    errors = []

    for entity in entities:
        try:
            data = kg_manager.query_full(entity)
            if not data or not data.get("entity_type"):
                not_found.append(entity)
                continue
            for key in ("spots", "festivals", "categories", "nearby_spots",
                        "spots_in_area", "cities", "prefectures"):
                if key in data:
                    data[key] = _dedup(data[key])
            results[entity] = data
        except Exception as e:
            error_msg = f"Errore per '{entity}': {str(e)}"
            print(f"  [KG Tool]  {error_msg}")
            errors.append(error_msg)

    if not results:
        return json.dumps({
            "status": "no_data",
            "message": ("Nessuna di queste entita' e' presente nel Knowledge Graph. "
                        "Affidati ai tool Web e RAG per reperire le informazioni."),
            "entities_not_found": not_found,
            "errors": errors if errors else None
        }, ensure_ascii=False, default=str)

    return json.dumps({
        "status": "success",
        "message": "Collegamenti trovati nel Knowledge Graph.",
        "krag_context": results,
        "entities_not_found": not_found if not_found else None,
        "errors": errors if errors else None
    }, ensure_ascii=False, indent=2, default=str)

@tool
def rag_retrieval_tool(query: str) -> str:
    """Usa questo tool per cercare nell'archivio locale.
    Contiene descrizioni di attrazioni, informazioni storiche e culturali sul Giappone.
    La query viene automaticamente ESPANSA o RAFFINATA usando il Knowledge Graph (K-RAG)
    per migliorare la pertinenza dei documenti recuperati."""
    print(f"[Esecuzione Tool] K-RAG (Chroma+BM25 + espansione KG) per: {query}")

    docs, exp = rag_manager.krag_search(query, k=8)

    header = ""
    if exp.get("seed_entities"):
        header = (
            "[K-RAG] Query guidata dal Knowledge Graph.\n"
            f"Modalita': {exp.get('mode')}\n"
            f"Entita' seed dal KG: {', '.join(exp['seed_entities'])}\n"
            f"Termini di espansione: {', '.join(exp['expansion_terms']) or 'nessuno (query logistica/segmentata)'}\n"
            f"Sotto-query eseguite: {exp['subqueries']}\n\n"
        )
    else:
        header = "[K-RAG] Nessuna entita' del grafo trovata nella query: retrieval sulla sola query originale.\n\n"




    return header + KRAG_GROUNDING_DELIM + rag_manager.format_search_results(docs)



geolocator = Nominatim(user_agent="japan_travel_blogger_agent")

@lru_cache(maxsize=128)
def geocode_location(location_name: str):
    """
    Tenta di geocodificare una località. Usa una cache LRU per evitare di 
    chiamare le API di Nominatim due volte per la stessa città.
    """
    time.sleep(1) 
    try:
        query = f"{location_name}, Giappone"
        location = geolocator.geocode(query, timeout=5)
        if location:
            return (location.latitude, location.longitude)
        
        location_fallback = geolocator.geocode(location_name, timeout=5)
        if location_fallback:
             return (location_fallback.latitude, location_fallback.longitude)
        
        return None
        
    except GeocoderTimedOut:
        print(f"  [Route Tool] Timeout API per: {location_name}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcola la distanza in km tra due coordinate (Haversine)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

@tool(args_schema=TravelRouteInput)
def travel_route_tool(locations_list: list[str]) -> str:
    """
    Calcola l'itinerario ottimale (Nearest Neighbor TSP), le distanze in km e i tempi di percorrenza 
    stimati (in treno/Shinkansen) data una lista di città o attrazioni (es. ["Tokyo", "Kyoto", "Osaka"]).
    Usa questo tool per ottenere dati reali e logistici per la sezione "Come muoversi" o per pianificare gli itinerari.
    Args:
        locations_list: Lista di nomi di città, attrazioni o prefetture in Giappone.
                       Esempio: ["Tokyo", "Kyoto", "Osaka"]
    
    Returns:
        Una stringa JSON con l'itinerario ottimizzato e i dettagli delle tappe.
    """
    print(f"[Route Tool] Calcolo itinerario ottimale per: {locations_list}")
    
    if len(locations_list) < 2:
        return "Servono almeno due località per calcolare un itinerario."

    
    coords = {}
    ignored = []
    for loc in locations_list:
        coord = geocode_location(loc)
        if coord:
            coords[loc] = coord
        else:
            ignored.append(loc)
            print(f"  [Route Tool] Località ignorata (non geocodificata): {loc}")

    
    if len(coords) < 2:
        if ignored:
            suggerimento = "Suggerisco di usare nomi più precisi in inglese (es. 'Kyoto' invece di 'Kioto')."
        else:
            suggerimento = "Verifica che i nomi siano corretti e riprova."
        return json.dumps({
            "errore": "Non ho trovato abbastanza coordinate valide.",
            "località_non_trovate": ignored,
            "suggerimento": suggerimento
        }, ensure_ascii=False, indent=2)

    unvisited = list(coords.keys())
    current_node = unvisited.pop(0)
    optimized_route = [current_node]
    total_distance = 0.0
    route_details = []
    while unvisited:
        lat1, lon1 = coords[current_node]
        nearest_node = None
        min_dist = float('inf')
        for candidate in unvisited:
            lat2, lon2 = coords[candidate]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = candidate
        min_dist *= 1.15
        if min_dist > 50:
            tempo_ore = min_dist / 250.0  # Shinkansen
            mezzo = "Treno ad alta velocità (Shinkansen)"
        elif min_dist > 20:
            tempo_ore = min_dist / 100.0  # Espresso regionale
            mezzo = "Treno espresso"
        else:
            tempo_ore = min_dist / 30.0   # Treno locale/metro
            mezzo = "Treno locale / Metro"

        ore = int(tempo_ore)
        minuti = int((tempo_ore - ore) * 60)
        tempo_str = f"{ore}h {minuti}m" if ore > 0 else f"{minuti} minuti"

        route_details.append({
            "da": current_node,
            "a": nearest_node,
            "distanza_km": round(min_dist, 1),
            "mezzo_consigliato": mezzo,
            "tempo_stimato": tempo_str
        })

        total_distance += min_dist
        optimized_route.append(nearest_node)
        unvisited.remove(nearest_node)
        current_node = nearest_node

    result = {
        "itinerario_ottimizzato": " → ".join(optimized_route),
        "distanza_totale_km": round(total_distance, 1),
        "dettagli_tappe": route_details
    }

    if ignored:
        result["avviso"] = f"Alcune località non sono state geocodificate e sono state ignorate: {', '.join(ignored)}. Per risultati migliori, usa i nomi ufficiali in inglese."

    return json.dumps(result, ensure_ascii=False, indent=2)

_fc_model = None
_fc_tokenizer = None

def get_fact_checker():
    """Funzione che carica il nostro modello LoRA come una mamma prepara la merenda: una volta sola!"""
    global _fc_model, _fc_tokenizer
    if _fc_model is None:
        print("[Fact Checker] Sveglio il modello LoRA...")
        model_path = "./model/fact_checker_lora"
        
        
        _fc_tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", num_labels=3)
        _fc_model = PeftModel.from_pretrained(base_model, model_path)
        _fc_model.eval()
    
    return _fc_model, _fc_tokenizer

def evaluate_claim_with_lora(context: str, claim: str) -> str:
    """Questa è la lente d'ingrandimento: guarda contesto e claim e ci dà il voto."""
    if not context or not claim:
        return "NEUTRO (Input vuoto o non valido)"
    
    model, tokenizer = get_fact_checker()
    
    inputs = tokenizer(context, claim, return_tensors="pt", truncation="only_first", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        
    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    
    id2label = {0: "CONTRADDIZIONE (Falso)", 1: "ENTAILMENT (Vero)", 2: "NEUTRO (Non ci sono info sufficienti)"}
    return id2label[predicted_class_id]

@tool
def fact_checking_tool(context: str, claim: str) -> str:
    """
    Usa questo tool per verificare la veridicità di un'affermazione (claim) rispetto a un testo di riferimento (context).
    Restituisce un giudizio formale basato su un modello LLM addestrato (LoRA): ENTAILMENT (Vero), CONTRADDIZIONE (Falso) o NEUTRO.
    Usalo ogni volta che devi validare un fatto dubbio sui dati turistici.
    """
    return evaluate_claim_with_lora(context, claim)


tools_list = [advanced_web_research, kg_query_tool, rag_retrieval_tool, travel_route_tool, fact_checking_tool]