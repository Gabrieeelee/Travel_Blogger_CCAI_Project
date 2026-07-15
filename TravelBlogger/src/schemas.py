from pydantic import BaseModel, Field
from typing import List, Optional

class PostPlan(BaseModel):
    topic: str = Field(description="La meta, l'attrazione o la struttura specifica. Se è un itinerario, separa rigorosamente le città con virgola (es. 'Tokyo, Kyoto, Osaka'). Se è una recensione ('review'), DEVI inserire il NOME ESATTO e specifico della struttura (es. 'Park Hyatt Tokyo', 'Ichiran Ramen Shibuya'), NON inserire mai la città generica.")
    type: str = Field(description="DEVE essere uno tra: 'itinerary' (viaggio multi-tappa), 'review' (recensione di un singolo hotel/ristorante/onsen), o 'guide' (guida generale su una città o prefettura).")
    angle: str = Field(description="Il taglio editoriale o la prospettiva specifica dell'articolo (es. 'Budget low-cost', 'Miglior street food').")
    duration: str = Field(default="", description="La durata del viaggio se specificata dall'utente (es. '4 giorni', '1 settimana').")

class EditorialCalendar(BaseModel):
    reasoning_process: str = Field(description="Un messaggio diretto all'utente in cui spieghi e giustifichi l'ordine e la selezione dei topic scelti.")
    sequence: List[PostPlan] = Field(description="La sequenza dei post da pubblicare. REGOLA CRITICA: Genera 1 SOLO POST se l'utente chiede un itinerario (itinerary) o una recensione specifica (review). Genera 3 POST se l'utente chiede suggerimenti generici o guide standard (guide).")

class PlannerIntent(BaseModel):
    mode: str = Field(description="'confirm_active' se l'utente accetta di procedere con il piano in sospeso (es. dice 'ok', 'vai avanti'), 'new_plan' se chiede un argomento/meta nuova, 'modify_plan' se chiede modifiche all'itinerario proposto (es. 'cambia la terza tappa', 'tieni i primi due').")

class PlanApprovalRouting(BaseModel):
    decision: str = Field(description="Classifica la decisione dell'utente: 'approve' se conferma o accetta il piano, 'modify' se rifiuta, chiede modifiche o propone un cambio di meta turistica.")

class TravelRouteInput(BaseModel):
    locations_list: List[str] = Field(
        description="Lista di nomi di città, attrazioni o prefetture in Giappone.",
        examples=[["Tokyo", "Kyoto", "Osaka"], ["Sapporo", "Hakodate", "Noboribetsu"]]
    )

class LocationMapping(BaseModel):
    spot: str = Field(description="Nome dell'attrazione o punto di interesse (lascia vuoto se non applicabile)")
    city: str = Field(description="Città esatta in cui si trova")
    prefecture: str = Field(description="Prefettura esatta di appartenenza")

class TravelMetadataExtractor(BaseModel):
    mapped_locations: List[LocationMapping] = Field(description="Mappatura esatta tra attrazione, città e prefettura.")
    category: List[str] = Field(default=[], description="Categorie dell'itinerario o dell'attrazione (es. Shrine, Temple, Modern/City, Nature/Hike).")
    local_foods: List[str] = Field(default=[], description="Cibi o piatti locali menzionati nel testo.")
    crafts: List[str] = Field(default=[], description="Artigianato tipico menzionato.")
    sources: List[str] = Field(default=[], description="Lista di URL delle fonti citate nel dossier.")
    claims: List[str] = Field(default=[], description="Lista di 2-3 fatti chiave trattati.")

class FeedbackRouting(BaseModel):
    decision: str = Field(description="Una tra: 'need_research', 'change_topic', 'rewrite', 'approve'")
    reasoning: str = Field(description="Spiegazione logica del perché è stata presa questa decisione")

class BaseDossier(BaseModel):
    title: str = Field(description="Titolo accattivante del dossier turistico/itinerario/recensione.")
    introduction: str = Field(description="Sintesi discorsiva e dettagliata del focus editoriale (angle). Almeno 3 frasi.")
    fact_checks: List[str] = Field(default_factory=list, description="Avvisi su incongruenze tra le fonti.")
    sources: List[str] = Field(default_factory=list, description="Elenco crudo di TUTTI gli URL consultati.")

class SourcedSection(BaseModel):
    text: str = Field(description="Testo lungo, descrittivo, discorsivo e ricco di dettagli narrativi.")
    source_urls: List[str] = Field(default_factory=list, description="Lista esatta degli URL (estratti dai tool grezzi) usati come fonte per redigere questo testo. Inserisci SOLO URL realmente presenti nei dati.")

class AttractionDetail(BaseModel):
    name: str = Field(description="Nome esatto dell'attrazione, tempio, parco o quartiere.")
    description: str = Field(description="Descrizione dettagliata e narrativa dell'attrazione.")
    source_url: str = Field(description="L'URL ESATTO da cui hai estratto questa informazione. Nessuna allucinazione consentita. Se l'info deriva dal KNOWLEDGE GRAPH, scrivi 'Knowledge Graph'.")

class PracticalInfoDetail(BaseModel):
    detail: str = Field(description="Il dato pratico specifico (es. 'Il biglietto per la metro costa 200 yen', 'Chiuso il martedì').")
    source_url: str = Field(description="L'URL ESATTO da cui hai estratto questo dato numerico/pratico.")

class ResearchDossier(BaseDossier):
    history_culture: Optional[SourcedSection] = Field(description="Contesto storico e cenni culturali.")
    logistics: Optional[SourcedSection] = Field(default=None, description="Testo discorsivo, lungo e dettagliato sulle info pratiche (distanze in km, trasporti, treni, aeroporti).")
    food_crafts: Optional[SourcedSection] = Field(description="Piatti tipici e artigianato. Se non trovi info, scrivi 'Nessuna informazione specifica' in text.")
    practical_info: List[PracticalInfoDetail] = Field(default_factory=list, description="Dati pratici ESTREMAMENTE SPECIFICI (prezzi, orari).")
    attractions: List[AttractionDetail] = Field(default_factory=list, description="Elenco dettagliato delle attrazioni e delle cose da vedere.")

class ClaimInfo(BaseModel):
    claim: str = Field(description="L'affermazione fattuale esatta da verificare (deve contenere dati, numeri, entità o fatti storici/logistici verificabili). Evita le opinioni.")
    paragrafo_contesto: str = Field(description="Il paragrafo o la frase ESATTA tratta dal testo originale in cui appare questo claim. Questo servirà al modello per la verifica.")

class ExtractedClaims(BaseModel):
    lista_claims: List[ClaimInfo] = Field(default_factory=list, description="Tutte le affermazioni fattuali più importanti trovate nel testo.")

class DayPlan(BaseModel):
    day_title: str = Field(description="Titolo della giornata (es. 'Giorno 1: Arrivo a Tokyo e Shibuya').")
    description: str = Field(description="Descrizione narrativa e dettagliata delle attività.")
    logistics: str = Field(description="Dettagli fondamentali sugli spostamenti (distanze, tempi dei treni, mezzi consigliati).")
    source_urls: List[str] = Field(default_factory=list, description="Lista degli URL usati come fonte.")

class ItineraryDossier(BaseDossier):
    days: List[DayPlan] = Field(description="Elenco dettagliato giorno per giorno dell'itinerario.")
    practical_info: List[PracticalInfoDetail] = Field(default_factory=list, description="Consigli pratici universali sul viaggio (es. Japan Rail Pass, SIM card).")

class ReviewDossier(BaseDossier):
    # Eredita: title, introduction, fact_checks, sources
    facility_name: str = Field(description="Nome esatto della struttura, hotel o ristorante.")
    pros_and_cons: SourcedSection = Field(description="Analisi critica: pro e contro emersi dalle recensioni e dalle fonti.")
    pricing_and_booking: PracticalInfoDetail = Field(description="Costi medi, ticket o prezzi del menu.")
    verdict: SourcedSection = Field(description="Giudizio critico finale.")




