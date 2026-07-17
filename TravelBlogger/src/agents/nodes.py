import os
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, BaseMessage, AIMessage,ToolMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.schemas import EditorialCalendar, ItineraryDossier, PlannerIntent, PlanApprovalRouting,FeedbackRouting, ResearchDossier, ReviewDossier, TravelMetadataExtractor, ExtractedClaims, ClaimInfo
from src.state import BloggerState
from langgraph.types import Command, interrupt
from src.tools.tools import kg_query_tool,advanced_web_research,rag_retrieval_tool, tools_list, evaluate_claim_with_lora
from typing import Dict, List, Any, Literal
from src.kg_manager import kg_manager
from src.rag_manager import rag_manager
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from datetime import datetime
from typing import Dict
import json
from typing import Literal
from langgraph.graph import StateGraph, START, END
import re

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2 
)
llm_with_tools = llm.bind_tools(tools_list)


# ═══════════════════════════════════════════
# 1. NODO PLANNER 
# ═══════════════════════════════════════════

def planner_node(state: BloggerState):
    print("\n--- [PlannerNode] Analisi e Pianificazione Travel ---")
    user_input = state.get("user_input", "")
    
    
    current_month = datetime.now().month
    coverage_gaps = kg_manager.get_coverage_gaps()
    theme_gaps = kg_manager.get_theme_coverage_gaps(limit=3)
    upcoming_festivals = kg_manager.get_upcoming_festivals(month=current_month)
    recent_posts = kg_manager.get_recent_posts(limit=20)
    
    travel_context = f"""
    ANALISI COPERTURA (Prefetture con MENO articoli, da privilegiare): {coverage_gaps}
    FESTIVAL DEL MESE IN CORSO (Mese {current_month}): {upcoming_festivals}
    TEMI IN EVIDENZA (Da coprire): {theme_gaps}
    POST RECENTI GIA' PUBBLICATI (Da NON ripetere): {recent_posts}
    """
    
    active_plan = kg_manager.get_active_plan_status()
    mode = "new_plan" 
    
    # ==========================================
    # 1. VERIFICA INTENZIONI UTENTE
    # ==========================================

    if active_plan:
        print("[Planner] Trovato calendario editoriale attivo nel database.")
        
        intent_llm = llm.with_structured_output(PlannerIntent)
        intent_prompt = f"""C'è un piano editoriale di viaggio in sospeso: {str(active_plan)}
        L'utente ha scritto: '{user_input}'
        Sta confermando di voler procedere con il piano (es. 'ok', 'sì', 'procedi'), o sta chiedendo una meta nuova/diversa?"""
        
        intent_result = intent_llm.invoke([
            SystemMessage(content="Sei un analista di intenti. Classifica la richiesta dell'utente."),
            HumanMessage(content=intent_prompt)
        ])
        mode = intent_result.mode
        print(f"[Planner] Intento rilevato: {mode}")


    delete_msgs = [RemoveMessage(id=m.id) for m in state.get("messages", []) if getattr(m, 'id', None)]
    
    reset_state = {
        "human_feedback": "",  
        "kg_summary": "",
        "action_results": {},
        "reasoning_trace": [],
        "research_summary": "",
        "fact_check_report": "",
        "current_draft": "",
        "tool_call_count": 0,
        "messages": delete_msgs
        
    }
    # ==========================================
    # CONFERMA PIANO ESISTENTE
    # ==========================================

    if mode == "confirm_active" and active_plan:
        print("\n Piano precedente confermato. Salto la generazione e l'approvazione.")
        topic_of_the_day = active_plan[0]
        print(f" META DEL GIORNO SELEZIONATA: {topic_of_the_day['topic']}\n")
        
        return {
            "editorial_plan": [topic_of_the_day], 
            "full_calendar": active_plan,
            "human_feedback": "",  
            **reset_state
        }

    # ==========================================
    # MODIFICA PIANO ESISTENTE
    # ==========================================

    if mode == "modify_plan" and active_plan:
        print("\n L'utente ha chiesto modifiche al piano esistente. Procedo con il contesto aggiornato.")
        travel_context += f"\n\nPiano editoriale in sospeso:\n{str(active_plan)}"
        user_input += " (L'utente ha chiesto modifiche al piano precedente, usa questo feedback per generare un nuovo itinerario/piano.)"

    # ==========================================
    # GENERAZIONE NUOVO PIANO E MODIFICA
    # ==========================================

    print("\n[Planner] Creazione di un NUOVO piano editoriale Travel...")
    planner_llm = llm.with_structured_output(EditorialCalendar)
    
    system_prompt = f"""Sei l'Editor in Chief di una prestigiosa guida turistica sul Giappone.
    Devi generare un calendario editoriale in base alla richiesta dell'utente.

    REGOLE FONDAMENTALI (IN ORDINE DI PRIORITÀ):

    1. CLASSIFICAZIONE DELL'INTENTO E NUMERO DI POST (CRITICO E MATEMATICO):
    - [CASO 1 POST]: Se l'utente chiede ESPLICITAMENTE un "itinerario" che comprende più città/tappe (es. "itinerario di 5 città") OPPURE "UNA recensione" specifica di un hotel, ristorante o attrazione fisica, devi generare **ESATTAMENTE 1 POST** nella 'sequence'. 
        - Il campo 'type' sarà 'itinerary' o 'review'.
        - Se 'itinerary', il 'topic' DEVE contenere tutte le città separate da virgola.
        - Se 'review' e l'utente è stato generico (es. "recensione di un hotel a Tokyo"), DEVI scegliere TU una struttura fisica reale e famosa, e inserire il SUO NOME ESATTO nel campo 'topic' (es. "Aman Tokyo" o "Hotel Gracery Shinjuku"), NON inserire la città generica!
    - [CASO 3 POST]: Se l'utente usa il plurale (es. "scrivi degli articoli su Nagasaki", "consigliami delle mete") o chiede guide generiche su una singola città senza un itinerario fisso, DEVI GENERARE TASSATIVAMENTE **ESATTAMENTE 3 POST** nella 'sequence'. 
        - Imposta il 'type' su 'guide' per tutti e 3 i post.
        - Assegna a ciascuno un 'angle' COMPLETAMENTE DIVERSO (es. 1. Storia e Cultura, 2. Gastronomia Tipica, 3. Attrazioni Nascoste) per coprire la stessa meta da tre prospettive differenti.

    2. GESTIONE DUPLICATI CON LO STORICO RECENTE (CRITICO):
    - VERIFICA SEMPRE i "POST RECENTI GIÀ PUBBLICATI" forniti in {travel_context}.
    - **SOLO SE** la città richiesta (o le città) è effettivamente presente in quella lista, **NON** proporre lo stesso 'angle' (angolo/approccio). Costringiti a usare angoli completamente diversi (es. cibo locale, shopping, arte, eventi stagionali, ecc.).
    - Se la città **NON** è nella lista dei post recenti, sei libero di proporre gli angle che ritieni migliori (inclusa una guida generale).
    ATTENZIONE ALLA SEMANTICA: È severamente vietato usare angoli che significano la stessa cosa ma con parole diverse. Se nello storico esiste un post con angle "Guida storico-culturale", NON PUOI proporre "Storia e Cultura", "Storia", o "Tradizioni". Costringiti a usare angoli completamente inesplorati (es. vita notturna, percorsi naturalistici, shopping).

    3. GIUSTIFICAZIONE DELLE SCELTE (reasoning_process):
    - Compila il campo 'reasoning_process' rivolgendoti direttamente all'utente (es. "Ho scelto queste mete perché...", "Ho deciso di concentrarmi su questo tema in quanto...").
    - **SOLO SE** hai rilevato un duplicato reale nei "POST RECENTI", usa questo campo per avvertire l'utente (es. "Attenzione: abbiamo già trattato questa città di recente, quindi ti propongo articoli con focus diversi").

    4. DIVERSIFICAZIONE GEOGRAFICA:
    - Applicala **SOLO** se la richiesta dell'utente è vaga e NON specifica una città o una regione precisa. In quel caso, distribuisci i post su aree geografiche differenti del Giappone.

    5. STAGIONALITÀ:
    - Se pertinente, includi un festival o un evento del mese in corso per arricchire i contenuti.

    6. RISPETTO DELLE REGOLE DELL'AGENTE:
    - Non generare mai duplicati.
    - Non ignorare le richieste dell'utente.
    - Non proporre mete già trattate nello storico recente senza variare gli angoli.
    - Non proporre mete fuori dal Giappone.

    DATI EDITORIALI ATTUALI E STORICO RECENTE:
    {travel_context}"""
    
    plan_result = planner_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Richiesta dell'utente: '{user_input}'")
    ])
    
    sequence = [{"topic": p.topic, "type": p.type, "angle": p.angle, "duration": getattr(p, "duration", "")} for p in plan_result.sequence]
    ragionamento = plan_result.reasoning_process
    
    feedback_history = ""
    
    while True:
        print("\n[Planner] Attendo la conferma o il feedback dell'utente sul piano proposto...")
        user_decision = interrupt({
            "type": "calendar_approval",
            "proposed_sequence": sequence,
            "agent_reasoning": ragionamento,
        })
        
        user_response = str(user_decision).strip()
        print(f"\n[Planner] Valutazione semantica della risposta utente: '{user_response}'...")
        
        approval_llm = llm.with_structured_output(PlanApprovalRouting)
        approval_result = approval_llm.invoke([
            ("system", "Sei un analista. Il tuo compito è classificare la risposta dell'utente alla proposta di un piano di viaggio. Se l'utente accetta, acconsente o dà una conferma generica (es. 'ok', 'va bene', 'perfetto', 'andiamo'), restituisci 'approve'. Se l'utente chiede di cambiare meta, propone altre città, o rifiuta, restituisci 'modify'."),
            ("user", f"Risposta dell'utente: '{user_response}'")
        ])
        
        if approval_result.decision == "modify":
            print(f"\n[Planner] Rielaborazione del piano in corso... (L'utente ha chiesto modifiche)")
            
    
            feedback_history += f"\n- L'utente ha esplicitamente richiesto: '{user_response}'"
            updated_travel_context = travel_context + f"\n\nCRONOLOGIA FEEDBACK (Direttive assolute da seguire ORA):\n{feedback_history}"
            feedback_system_prompt = f"""Sei il Direttore Editoriale Travel. Devi rigenerare il calendario di 3 mete.
            
            REGOLE FONDAMENTALI (CRITICHE):
            1. CAMBIO RADICALE (ASCOLTA L'UTENTE): Se nella "CRONOLOGIA FEEDBACK" l'utente ti chiede di cambiare meta, DEVI ABBANDONARE il piano precedente e i gap di copertura generali. Obbedisci ciecamente generando un nuovo piano sulla NUOVA META indicata.
            2. GIUSTIFICAZIONE AGGIORNATA (reasoning_process): Compila 'reasoning_process' per confermare all'utente che hai recepito la sua modifica (es. "Come da te richiesto, ho scartato la proposta precedente e ho creato un itinerario su...").
            3. CONSERVAZIONE PARZIALE: Se l'utente chiede di modificare SOLO un aspetto (es. "il terzo articolo non mi piace"), mantieni i primi due inalterati e cambia solo il terzo.
            4. GESTIONE DUPLICATI: VERIFICA SEMPRE I "POST RECENTI GIA' PUBBLICATI". SOLO SE la nuova meta scelta è effettivamente già coperta in quella lista, mantieni la meta ma varia drasticamente gli angoli. È vietato inventare post passati inesistenti.
            5. DIVERSIFICAZIONE GEOGRAFICA: Applicala SOLO se la richiesta dell'utente è vaga e NON specifica una città o regione precisa.
            6. STAGIONALITÀ: Se pertinente, includi un festival del mese in corso.
            7. RISPETTO DELLE REGOLE DELL'AGENTE: Non generare mai duplicati, non ignorare le richieste dell'utente, non proporre mete già trattate nello storico recente senza variare gli angoli, e non proporre mete fuori dal Giappone.
            
            DATI EDITORIALI ATTUALI E STORICO RECENTE:
            {updated_travel_context}"""
            
            feedback_user_prompt = f"Richiesta originale: '{user_input}'\nPiano appena rifiutato:\n{str(sequence)}\n\n💬 NUOVO FEEDBACK DA APPLICARE IMMEDIATAMENTE: '{user_response}'"
            travel_context = updated_travel_context
            new_plan_result = planner_llm.invoke([
                SystemMessage(content=feedback_system_prompt),
                HumanMessage(content=feedback_user_prompt)
            ])
            
            sequence = [{"topic": p.topic, "type": p.type, "angle": p.angle} for p in new_plan_result.sequence]
            ragionamento = new_plan_result.reasoning_process 
        else:
            print("\n Piano approvato dall'utente. Uscita dal ciclo di revisione.")
            break

    kg_manager.save_active_plan(sequence)

    topic_of_the_day = sequence[0]
    print(f"\n META DEL GIORNO SELEZIONATA: {topic_of_the_day['topic']}\n")
    
    return {
        "editorial_plan": [topic_of_the_day], 
        "full_calendar": sequence,
        "human_feedback": "",  
        **reset_state   
    }

# ═══════════════════════════════════════════
# 2. NODO RESEARCH
# ═══════════════════════════════════════════

def research_warmup_node(state: BloggerState):
    print("\n[Research Init] Estrazione deterministica dal Knowledge Graph...")
    
    plan = state.get("editorial_plan", [])
    topic = plan[0]["topic"] if plan else state.get("user_input", "Giappone")
    topic_type = plan[0].get("type", "guide")
    angle = plan[0].get("angle", "Generale")
    feedback = state.get("human_feedback", "")
    action_results = state.get("action_results", {})
    
    if not action_results:
        action_results = {}

    reasoning = state.get("reasoning_trace", [])
    kg_summary_state = state.get("kg_summary", "")
    
    if feedback and kg_summary_state:
        print("[Warmup] Feedback e dati KG presenti, salto l'estrazione KG base.")
        kg_res = kg_summary_state
    else:
        print(f"[Warmup] Interrogazione KG per: '{topic}'")
        try:
            kg_res = str(kg_query_tool.invoke(topic))
            obs_kg = kg_res[:350]
        except Exception as e:
            kg_res = f"Errore KG: {e}"
            obs_kg = kg_res
            
        action_results.setdefault("kg_query_tool", []).append(kg_res)
        
        reasoning.append({
            "agent": "researcher_warmup",
            "thought": f"Pre-caricamento del Knowledge Graph per '{topic}' per guidare le espansioni K-RAG.",
            "action": f"kg_query_tool('{topic}')",
            "observation": obs_kg
        })

    if topic_type == "itinerary":
        locations = [loc.strip() for loc in topic.split(",")]
        format_instruction = (
            f"Stai pianificando un ITINERARIO MULTI-TAPPA per {len(locations)} destinazioni: {', '.join(locations)}. "
            f"DEVI chiamare i tool di ricerca (web e RAG) SEPARATAMENTE per ogni singola tappa. "
            f"La tua priorità assoluta è usare il tool 'travel_route_tool' fornendogli l'intera lista delle tappe per calcolare i tempi di spostamento e le distanze."
        )
    elif topic_type == "review":
        format_instruction = (
            f"Stai scrivendo una RECENSIONE SPECIFICA su una singola struttura o luogo: '{topic}'. "
            f"Usa i tool per cercare informazioni iper-verticali: recensioni degli utenti, prezzi esatti, pro e contro, e servizi specifici offerti. Non divagare sulla storia della città."
        )
    else:  
        format_instruction = (
            f"Stai scrivendo una GUIDA TURISTICA generale su: '{topic}'. "
            f"Cerca informazioni ampie su storia, cultura, principali attrazioni (costi/orari) e consigli pratici generali."
        )
    system_prompt = f"""Sei il Lead Researcher per un'autorevole Guida Turistica sul Giappone.
    Il tuo OBIETTIVO è raccogliere informazioni da passare ai copywriter per scrivere un post perfetto.
    
    DIRETTIVE EDITORIALI:
    - Meta/Mete: '{topic}'
    - Formato Articolo: '{topic_type.upper()}'
    - Focus Editoriale (Angle): '{angle}'
    - Feedback del Direttore (se presente): {feedback}

    ISTRUZIONI SPECIFICHE PER QUESTO FORMATO (CRITICO):
    {format_instruction}

    DATI DI PARTENZA (KNOWLEDGE GRAPH):
    Ho già interrogato il Knowledge Graph principale per te. Leggi attentamente cosa sappiamo già:
    {kg_res[:1500]}

    ISTRUZIONI OPERATIVE E STRATEGIA DEI TOOL (ReAct):
    Ora tocca a te. Usa i tool a tua disposizione in modo dinamico per raccogliere i dati mancanti e coprire in profondità l'angle '{angle}'.

    0. PENSA AD ALTA VOCE (CRITICO): Prima di invocare QUALSIASI tool, devi SEMPRE scrivere una breve frase discorsiva (Thought) per spiegare il tuo ragionamento. Devi dichiarare esplicitamente PERCHÉ hai scelto proprio quel tool. Non lanciare mai un tool in silenzio!

    1. RICERCA DI DATI PRATICI UNIVERSALI (CRITICO): I lettori odiano gli articoli generici. Per QUALUNQUE entità tu stia ricercando (Ristoranti, Hotel, Templi, Musei, Quartieri, Trasporti), DEVI cercare attivamente dati numerici reali: 
       - Per le attrazioni: Costo esatto dei biglietti, orari di apertura/chiusura, giorni di riposo, durata media della visita.
       - Per il cibo/ristoranti: Fascia di prezzo (es. "3000-5000 yen"), piatti forti, necessità di prenotazione.
       - Per gli hotel/strutture: Tariffe medie per notte, orari check-in, servizi inclusi, distanza a piedi dalle stazioni.
       - Per i trasporti: Costo esatto del biglietto, validità dei pass (es. JR Pass), tempi di percorrenza.

    2. STRATEGIA DELLE QUERY (DIVIETO DI KEYWORD SOUP):
       È ASSOLUTAMENTE VIETATO inviare query sature con decine di concetti o città diverse (es. "Kyoto, Osaka, Nara, storia, prezzi, orari"). 
       3. REGOLA DEL FOCUS GEOGRAFICO (CRITICA):
       In OGNI SINGOLA QUERY che invii ai tool, DEVI inserire sempre la parola chiave della meta (es. "Tokyo"). È severamente vietato fare ricerche generiche come "orari attrazioni" o peggio, cercare altre prefetture (es. "Hokkaido").
       I database vettoriali e i motori di ricerca falliscono con query del genere. DEVI fare ricerche ATOMICHE, MIRATE e SEPARATE.
       - Sbagliato: rag_retrieval_tool("Kyoto, Osaka, storia, prezzi, orari")
       - Corretto 1 (Ricerca Storica): rag_retrieval_tool("Storia, architettura e leggende del tempio Kiyomizu-dera a Kyoto")
       - Corretto 2 (Ricerca Logistica): advanced_web_research("Costo biglietto e orario apertura Castello di Osaka")
       - CORRETTO 3: advanced_web_research("prezzo biglietto ingresso Museo della Pace Hiroshima")
       Fai una chiamata ai tool per ogni singola città o singola attrazione. Separa le ricerche!

    4. LOGISTICA E ITINERARI (travel_route_tool):
       Se '{topic}' è un itinerario con più città, DEVI usare questo tool per calcolare tempi e distanze dei treni.
       Se stai scrivendo una 'guide' sulla singola città, PUOI usarlo per dare un'idea degli spostamenti, ma DEVI LIMITARTI a un massimo di 3 o 4 attrazioni. È SEVERAMENTE VIETATO calcolare mega-itinerari cittadini inserendo 10+ tappe come se fosse un tour de force.
       ESEMPIO D'USO: travel_route_tool(locations_list=["Tokyo", "Kyoto", "Nara"])
    
    5. KNOWLEDGE GRAPH AGGIUNTIVO (kg_query_tool):
       Se durante le ricerche RAG/Web scopri un'attrazione minore interessante e vuoi verificare se ne abbiamo mai parlato nel blog, puoi usare questo tool liberamente.

    6. VERIFICA DEI FATTI - MODELLO FINE-TUNED (fact_checking_tool):
       Se trovi un'informazione ambigua, date discordanti o un fatto potenzialmente falso nei documenti, DEVI usare questo tool per valutare il claim tramite il nostro modello IA addestrato.
       [FIX 2] IMPORTANTE: Il tool richiede due parametri stringa. 
       ESEMPIO D'USO: fact_checking_tool(context="il testo del documento web che contiene il dato", claim="l'affermazione specifica da verificare, es. Il tempio chiude alle 18:00")
       
       COME COMPORTARSI DOPO IL RESPONSO DEL TOOL:
       - Se è "ENTAILMENT": Usa l'informazione con sicurezza.
       - Se è "CONTRADDIZIONE": SCARTA l'informazione o fai una nuova ricerca web per correggerla.
       - Se è "NEUTRO": Menziona l'informazione ma specifica ai copywriter che le fonti non sono certe.

    REGOLA DI STOP ASSOLUTA:
   Quando ritieni di aver raccolto dati testuali completi, esatti e verificati per coprire il focus '{angle}', SCRIVI semplicemente un testo descrittivo discorsivo in cui riassumi ai tuoi colleghi le tue scoperte.
    Generare un testo discorsivo senza usare i tool interromperà il tuo loop di ricerca."""
    
    new_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Inizia la tua ricerca autonoma. Usa i tool necessari per raccogliere le informazioni richieste.")
    ]

    return {
        "messages": new_messages,
        "action_results": action_results,
        "reasoning_trace": reasoning,
        "kg_summary": kg_res
    }

def research_agent_node(state: BloggerState):
    print("\n[Research Agent] Valutazione del contesto e scelta dell'azione...")
    messages = state.get("messages", [])
    
    if not messages:
        return research_warmup_node(state)

    temp_messages = list(messages)
    
    if messages and hasattr(messages[-1], 'type') and messages[-1].type == 'tool':
        ghost_reminder = HumanMessage(
            content="Valuta i risultati del tool qui sopra. Se hai raccolto tutte le informazioni necessarie per coprire il focus editoriale, scrivi direttamente un riassunto finale senza chiamare altri tool. Se mancano informazioni, chiama il prossimo tool necessario."
        )
        temp_messages.append(ghost_reminder)

    try:
        response = llm_with_tools.invoke(temp_messages)
    except Exception as e:
        print(f"\n[Research Agent] Errore API Groq recuperato: {e}")
        from langchain_core.messages import AIMessage
        response = AIMessage(content="Le API di ricerca hanno riscontrato un limite temporaneo. Procedo a generare il dossier con le informazioni raccolte finora.")

    return {"messages": [response]}

def execute_tools_node(state: BloggerState):
    print("[Execute Tools] Invocazione degli strumenti richiesti dall'LLM...")
    messages = state.get("messages", [])
    last_message = messages[-1] 
    tool_call_count = state.get("tool_call_count", 0)
    action_results = state.get("action_results", {})
    reasoning = state.get("reasoning_trace", [])
    tool_map = {t.name: t for t in tools_list}
    
    kg_summary_update = state.get("kg_summary", "")
    
    tool_messages = []
    
    agent_thought = last_message.content.strip() if last_message.content else "Scelta autonoma del tool basata sull'istruzione corrente."
    
    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        print(f"-> Loop Agentico: Eseguo {tool_name} | Pensiero: {agent_thought[:60]}...")
        
        try:
            tool_func = tool_map[tool_name]
            tool_result = str(tool_func.invoke(tool_args))
        except Exception as e:
            tool_result = f"Errore: {str(e)}"
            
        action_results.setdefault(tool_name, []).append(tool_result)
        
        if tool_name == "kg_query_tool":
            kg_summary_update += f"\n\n[NUOVI DATI KG ESTRATTI DINAMICAMENTE]:\n{tool_result}"
        
        reasoning.append({
            "agent": "researcher_agent",
            "thought": agent_thought,
            "action": f"{tool_name}({tool_args})",
            "observation": tool_result[:350]
        })
        
        tool_messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tc["id"]))
        tool_call_count += 1

    return {
        "messages": tool_messages,
        "action_results": action_results,
        "reasoning_trace": reasoning,
        "tool_call_count": tool_call_count,
        "kg_summary": kg_summary_update
    }

def route_research(state: BloggerState) -> Literal["execute_tools", "__end__"]:
    messages = state.get("messages", [])
    tool_call_count = state.get("tool_call_count", 0)
    if not messages:
        return "__end__"
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if tool_call_count >= 8:
            print(f"\n[Router] Raggiunto l'hard-limit di {tool_call_count} chiamate ai tool. Uscita forzata dal loop per prevenire crash.")
            return "__end__"
        
        return "execute_tools"
    return "__end__"

researcher_builder = StateGraph(BloggerState)

researcher_builder.add_node("warmup", research_warmup_node)
researcher_builder.add_node("research_agent", research_agent_node)
researcher_builder.add_node("execute_tools", execute_tools_node)

researcher_builder.add_edge(START, "warmup")           
researcher_builder.add_edge("warmup", "research_agent")  

researcher_builder.add_conditional_edges("research_agent", route_research)
researcher_builder.add_edge("execute_tools", "research_agent")

research_subgraph = researcher_builder.compile()

# ═══════════════════════════════════════════
# 3. NODO RECAP
# ═══════════════════════════════════════════

def recap(state: BloggerState):
    
    plan = state.get("editorial_plan", [])
    topic = plan[0]["topic"] 
    topic_type = plan[0].get("type", "guide")
    angle = plan[0].get("angle", "Generale") 
    duration = plan[0].get("duration", "")
    
    action_results = state.get("action_results", {})
    kg_summary = state.get("kg_summary", "")
    reasoning = state.get("reasoning_trace", []) 
    
    unique_outputs = set()
    raw_data_string = ""
    for tool_name, outputs in action_results.items():
        raw_data_string += f"\n ESPORTAZIONE TOOL: {tool_name} \n"
        for idx, out in enumerate(outputs):
            testo_pulito = str(out).strip()
            if testo_pulito not in unique_outputs:
                unique_outputs.add(testo_pulito)
                raw_data_string += f"[{idx+1}] {testo_pulito[:15000]}...\n"

    if topic_type == "itinerary":
        format_instructions = "- STRUTTURA: Itinerario multi-tappa. Struttura i dati cronologicamente per coprire più tappe, con sezioni distinte e dettagliate per ogni singola giornata o città."
    elif topic_type == "review":
        format_instructions = "- STRUTTURA: Recensione iper-verticale. Concentrati ESCLUSIVAMENTE sui dettagli iper-locali di una singola struttura (hotel, ristorante, attrazione). I campi 'attractions' devono diventare i servizi offerti (es. 'Piscina', 'Spa', 'Wi-Fi'), e 'practical_info' includerà prezzi esatti, orari di check-in/out e politiche interne."
    else:
        format_instructions = "- STRUTTURA: Guida generale. Struttura il dossier come un reportage ampio sulla destinazione, organizzando in modo chiaro le informazioni su storia, cultura, attrazioni principali e logistica generale."

    system_prompt = f"""Sei il Capo Analista per un'autorevole Guida Turistica sul Giappone.
    Devi distillare i frammenti grezzi estraendo i dati in una struttura JSON rigorosa.

    OBIETTIVO: Meta: '{topic}' | Formato: '{topic_type}' | Taglio (Angle): '{angle}'
    {f"VINCOLO TEMPORALE CRITICO: L'itinerario DEVE essere strutturato ESATTAMENTE su {duration}." if duration else ""}

    DATI GREZZI ESTRATTI (da tool web, RAG e ricerche):
    {raw_data_string}

    REGOLE CRITICHE PER LA DISTILLAZIONE (SEGUIRE IN ORDINE):

    1. **ADATTAMENTO DEL FORMATO AL TIPO DI POST ({topic_type.upper()})**:
    {format_instructions}

    2. **CAMPI OBBLIGATORI E STRUTTURA**:
    - **DIVIETO DI OMISSIONE**: Non omettere MAI chiavi previste dallo schema. Se per il focus dell'articolo non trovi informazioni per una specifica sezione, DEVI comunque generare la chiave inserendo "Nessuna informazione rilevante" e lasciando vuota la lista delle fonti.
    - `fact_checks`: popola questo campo se noti informazioni contrastanti tra le fonti; riporta brevemente la discrepanza e indica quale versione hai scelto.
    - `attractions` e `practical_info`: devono essere liste di oggetti esatti.
        - Per `attractions`: {{"name": "...", "description": "...", "source_url": "..."}}
        - Per `practical_info`: {{"detail": "...", "source_url": "..."}}

    3. **STRUTTURA RICCA**:
    Compila questi campi con le informazioni presenti nei dati grezzi, senza inventare nulla, se non sono presenti dati reali, salta il campo ma lascia la chiave vuota.:
    - ESTRAZIONE FEDELE E DIVIETO DI ALLUCINAZIONE: Estrai SOLO ciò che è scritto nei testi grezzi. 
    - `history_culture.text`: almeno 2-3 paragrafi con cronologia, eventi, personaggi
    - `attractions`: almeno 5-7 oggetti con nome, descrizione dettagliata, source_url
    - `practical_info`: almeno 5-7 dati numerici (prezzi, orari, distanze)
    - `logistics.text`: informazioni sui trasporti con dettagli (compagnie, costi, tempi)
    - Se NON ci sono dati logistici, prezzi, info pratiche o nomi di ristoranti nei documenti, LASCIA I CAMPI VUOTI [].
    - È SEVERAMENTE VIETATO inventare dati o URL per riempire lo schema.

    3. **RISOLUZIONE ALLUCINAZIONI SULLE FONTI (CRITICO)**:
    - **NON** scrivere mai manualmente "[Fonte: URL]" all'interno dei testi descrittivi (`text`, `description`, `detail`).
    - Usa **ESCLUSIVAMENTE** gli appositi campi `source_url` o `source_urls` previsti nei sottomodelli.
    - Ogni URL inserito DEVE essere esattamente quello da cui hai tratto l'informazione. Se metti l'URL di Tokyo su un dato di Osaka, fallirai il compito.
    - Se un'informazione proviene dal Knowledge Graph, scrivi **"Knowledge Graph"** nel campo `source_url`.

    4. **STILE NARRATIVO**:
    - Il campo `text` (o `description`) deve rimanere narrativo, lungo e discorsivo, ma **privo di link testuali** al suo interno.

    5. **ASSOCIAZIONE RIGOROSA (RICONTROLLO)**:
    - Verifica mentalmente: "Da quale fonte (URL o KG) proviene questo dato?" e assegna il `source_url` di conseguenza. 
    
    Ora, procedi con la distillazione. Restituisci esclusivamente il JSON strutturato secondo le specifiche sopra, senza ulteriori commenti.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Estrai il dossier strutturato e verificato per '{topic}' rispettando il taglio '{angle}'.")
    ]

    print("[Recap] Estrazione dei dati nel JSON Pydantic in corso...")
    
    if topic_type == "itinerary":
        structured_llm = llm.with_structured_output(ItineraryDossier)
    elif topic_type == "review":
        structured_llm = llm.with_structured_output(ReviewDossier)
    else:
        structured_llm = llm.with_structured_output(ResearchDossier)
    dossier_json = None
    
    for attempt in range(2):
        try:
            dossier_json = structured_llm.invoke(messages)
            if dossier_json.title: 
                break
        except Exception as e:
            print(f"[Recap] Errore al tentativo {attempt+1}: {e}")
            if attempt == 1:
                return {"research_summary": f"Errore di sintesi strutturata: {e}", "reasoning_trace": reasoning}

    
    def format_sourced_section(section):
        testo = section.text
        if section.source_urls:
            urls_str = ", ".join(section.source_urls)
            testo += f" [Fonti: {urls_str}]"
        else:
            testo += " [Fonte: Nessuna]"
        return testo

    # --- COMPILAZIONE MARKDOWN ---
    markdown_dossier = f"# {dossier_json.title}\n\n"
    markdown_dossier += f"## INTRODUZIONE E FOCUS EDITORIALE\n{dossier_json.introduction}\n\n"
    
    if topic_type == "itinerary":
        markdown_dossier += "## ITINERARIO GIORNO PER GIORNO\n"
        for day in dossier_json.days:
            fonti_day = ", ".join(day.source_urls) if day.source_urls else "Nessuna"
            markdown_dossier += f"### {day.day_title}\n"
            markdown_dossier += f"**Attività:** {day.description}\n"
            markdown_dossier += f"**Logistica:** {day.logistics} [Fonti: {fonti_day}]\n\n"
            
        if dossier_json.practical_info:
            markdown_dossier += "## INFO PRATICHE GENERALI\n"
            for info in dossier_json.practical_info:
                markdown_dossier += f"- {info.detail} [Fonte: {info.source_url}]\n"
            markdown_dossier += "\n"

    elif topic_type == "review":
        markdown_dossier += f"## ANALISI STRUTTURA: {dossier_json.facility_name}\n"
        markdown_dossier += f"**Prezzi e Prenotazioni:** {dossier_json.pricing_and_booking.detail} [Fonte: {dossier_json.pricing_and_booking.source_url}]\n\n"
        markdown_dossier += f"### Pro e Contro\n{format_sourced_section(dossier_json.pros_and_cons)}\n\n"
        markdown_dossier += f"### Giudizio Finale\n{format_sourced_section(dossier_json.verdict)}\n\n"

    else:
        if dossier_json.history_culture:   
            markdown_dossier += f"## STORIA E CULTURA\n{format_sourced_section(dossier_json.history_culture)}\n\n"
        
        if dossier_json.attractions:
            markdown_dossier += "## ATTRAZIONI E COSA VEDERE\n"
            for attr in dossier_json.attractions:
                markdown_dossier += f"- **{attr.name}**: {attr.description} [Fonte: {attr.source_url}]\n"

        if dossier_json.logistics:
            markdown_dossier += "\n"
            if dossier_json.logistics:
                markdown_dossier += f"## LOGISTICA E SPOSTAMENTI\n{format_sourced_section(dossier_json.logistics)}\n\n"

        if dossier_json.practical_info:
            markdown_dossier += "## INFO PRATICHE E PREZZI (DA INCLUDERE OBBLIGATORIAMENTE NEL TESTO)\n"
            for info in dossier_json.practical_info:
                markdown_dossier += f"- {info.detail} [Fonte: {info.source_url}]\n"
            markdown_dossier += "\n"

        if dossier_json.food_crafts:
            markdown_dossier += f"## CIBO E ARTIGIANATO\n{format_sourced_section(dossier_json.food_crafts)}\n\n"

    if dossier_json.fact_checks:
        markdown_dossier += "## FACT-CHECKING E SCARTI\n"
        for check in dossier_json.fact_checks:
            markdown_dossier += f"- {check}\n"
        markdown_dossier += "\n"
        
    markdown_dossier += "## FONTI CITATE\n"
    for src in dossier_json.sources:
        markdown_dossier += f"- {src}\n"

    reasoning.append({
        "agent": "recap",
        "thought": "Estrazione in schema Pydantic gerarchico completata. Ho isolato in sicurezza i source_url per evitare misattribution e convertito tutto in Markdown.",
        "action": "llm.with_structured_output(...)",
        "observation": markdown_dossier[:350]
    })
    
    print("[Recap] Dossier generato con successo dalla struttura JSON.")
    
    return {
        "research_summary": markdown_dossier,
        "reasoning_trace": reasoning
    }


def fact_checking_node(state: BloggerState):
    print("\n---  NODO: FACT CHECKING STRUTTURATO ---")
    
    riassunto = state.get("research_summary", "")
    
    if not riassunto:
        return Command(update={"fact_check_report": "Nessun riassunto da verificare."}, goto="drafter")
    
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ExtractedClaims)
    MAX_CLAIMS = 4
    system_prompt = f"""Sei un revisore editoriale specializzato in fact-checking per una Guida Turistica.
    Il tuo compito è estrarre TUTTE le affermazioni fattuali (claims) più importanti dal testo fornito.
    
    REGOLE CRITICHE:
    1. Ignora le opinioni, le descrizioni poetiche e le frasi generiche.
    2. Concentrati su dati verificabili: Date, fatti storici, prezzi, orari, logistica, nomi di luoghi e tradizioni.
    3. Per ogni claim, DEVI estrarre il 'paragrafo_contesto', ovvero la porzione esatta di testo da cui hai preso il claim. Non inventare il contesto, copialo.
    4. Se il testo contiene più di {MAX_CLAIMS} claim, estrai solo i primi {MAX_CLAIMS} più rilevanti.
    """
    
    print("[Fact Checker] Estrazione strutturata dei claim in corso...")
    
    try:
        risultato_estrazione = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Testo da analizzare:\n{riassunto}")
        ])
        claims_estratti = risultato_estrazione.lista_claims
    except Exception as e:
        print(f"    [Fact Checker] Errore nell'estrazione dei claim: {e}")
        claims_estratti = []
    
    report = "### Pagella del Fact-Checker Personale\n\n"
    
    if not claims_estratti:
        report += "Nessun claim fattuale rilevante trovato o errore di estrazione.\n"
    else:
        print(f" [Fact Checker] Trovati {len(claims_estratti)} claims fattuali. Valutazione LoRA in corso...")
        
        for item in claims_estratti:
            claim_testo = item.claim
            contesto_specifico = item.paragrafo_contesto
            
            giudizio = evaluate_claim_with_lora(context=contesto_specifico, claim=claim_testo)
            
            report += f"- **Affermazione:** {claim_testo}\n"
            report += f"  - **Contesto analizzato:** *{contesto_specifico}*\n"
            report += f"  - **Giudizio LoRA:** {giudizio}\n\n"
            
    print(report) 
    
    return {
        "fact_check_report": report
    }



# ═══════════════════════════════════════════
# 4. NODO DRAFTER
# ═══════════════════════════════════════════
def drafter_node(state: BloggerState):
    print("\n--- [DrafterNode] Stesura Articolo Travel Multimodale ---")
    
 
    creative_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2 
    )
    
    plan = state.get("editorial_plan", [])
    topic = plan[0]["topic"] if plan else state.get("user_input", "Giappone")
    topic_type = plan[0].get("type", "guide")
    angle = plan[0].get("angle", "Generale")
    duration = plan[0].get("duration", "")
    research_summary = state.get("research_summary", "Nessun dato di ricerca specifico disponibile. Basati sulla tua conoscenza generale turistica.")
    kg_summary = state.get("kg_summary", "Nessun dato storico.")
    human_feedback = state.get("human_feedback", "").strip()
    fact_check_report = state.get("fact_check_report", "Nessun controllo sui fatti disponibile.")
    reasoning = state.get("reasoning_trace", [])
    
    if topic_type == "itinerary":
        format_rules = f"""
    - STRUTTURA A GIORNATE: L'itinerario DEVE svilupparsi ESATTAMENTE su {duration if duration else 'le giornate indicate nel dossier'}. Scrivi un itinerario scandito giorno per giorno usando i tag H2 (es. '## Giorno 1: Esplorando Tokyo'). 
    - FOCUS LOGISTICO: Concentrati sul ritmo di viaggio. Integra fluidamente nel discorso i tempi degli spostamenti, i mezzi consigliati e i costi dei treni recuperati dal dossier.
    - TONO: Pratico, entusiasmante ma estremamente organizzato."""
    elif topic_type == "review":
        format_rules = """
    - STRUTTURA DA RECENSIONE: Usa un H2 (##) per la struttura/ristorante analizzato. Dividi il testo in sezioni chiare analizzando servizio, atmosfera e rapporto qualità/prezzo.
    - VERDETTO FINALE: L'articolo DEVE concludersi obbligatoriamente con un paragrafo H3 (###) intitolato 'Verdetto Finale', dove esprimi un'opinione netta (pro e contro).
    - TONO: Critico, valutativo e onesto. Sei un recensore severo ma giusto, non stai scrivendo una cartolina promozionale."""
    else: # guide
        format_rules = """
    - STRUTTURA NARRATIVA: Inizia con un Hook immersivo. Dividi l'articolo in paragrafi tematici (Storia, Attrazioni, Logistica) usando i tag H2. 
    - SHOW, DON'T TELL: Potenzia ulteriormente la descrizione sensoriale. Trasforma i dati freddi in esperienze vissute. Descrivi l'architettura, i sapori e l'atmosfera.
    - TONO: Evocativo, confidenziale ma autorevole. Trasporta il lettore fisicamente sul posto."""


    system_prompt = f"""Sei un Travel Blogger d'élite e un narratore viscerale, in stile reportage narrativo. Non scrivi voci enciclopediche o riassunti asettici: tu scrivi racconti di viaggio immersivi che trasportano il lettore fisicamente sul posto.
    Il tuo compito è scrivere l'articolo finale unendo ESCLUSIVAMENTE i dati del Dossier a uno storytelling coinvolgente e di altissima qualità.

    ### 1. REGOLA DI GROUNDING ASSOLUTO (CRITICO)
    - ZERO ALLUCINAZIONI: È SEVERAMENTE VIETATO introdurre eventi, festival, prezzi, cenni storici, tradizioni, informazioni logistiche (es. clima, moneta, lingua) o luoghi che NON siano esplicitamente scritti nel Dossier. 
    - L'unico input fattuale permesso è quello fornito sotto 'DOSSIER FATTUALE' e 'STORICO KNOWLEDGE GRAPH'.
    - Se il Dossier non contiene dettagli su un certo aspetto (es. la storia di un tempio), non menzionarlo o limitati a descriverne l'atmosfera visiva senza inventare date o fatti.

    ### 2. REGOLE DI STORYTELLING E STILE NARRATIVO (CRITICO)
    - HOOK INIZIALE: Non iniziare MAI con frasi banali come "[Nome Città] è una città che offre...". Inizia sempre l'articolo con una scena evocativa, un dettaglio visivo o un'azione.
    - SHOW, DON'T TELL (SENSORIALITÀ): Trasforma i dati freddi in esperienze vissute. Non dire che un tempio è "bello e antico"; descrivi il legno scuro consumato dal tempo, il suono delle campane tibetane o il sapore pungente del matcha.
    - ESPANSIONE NARRATIVA, NON FATTUALE: Per raggiungere la lunghezza desiderata, espandi le descrizioni sensoriali, le emozioni e le impressioni visive, NON l'elenco dei fatti. Gioca con le parole, non con i dati.
    - DIVIETO ASSOLUTO DI CLICHÉ: È severamente vietato usare espressioni banali come: "mix di antico e moderno", "tesoro nascosto", "città vibrante", "qualcosa per tutti".

    ### 3. STRUTTURA E FORMATTAZIONE GENERALE
    - Usa ESCLUSIVAMENTE il linguaggio Markdown. 
    - Inizia con un `# Titolo` narrativo e accattivante.
    - Dividi il testo usando `## Sottotitoli` descrittivi.
    - LOGISTICA E INFO PRATICHE: Se nel Dossier che ricevi è presente una sezione sulla Logistica e Spostamenti, incorporala come H2 ('## Come muoversi') raccontandola in modo evocativo ma preciso. Se il Dossier NON contiene informazioni logistiche, NON inventare nulla e ometti la sezione. Non fare MAI elenchi puntati asettici per i costi, intrecciali nel testo.

    ### 4. GUIDA DI STILE E STRUTTURA PER TIPOLOGIA DI ARTICOLO ({topic_type.upper()})
    {format_rules}

    ### 5. GESTIONE DEL FACT-CHECKING (CRITICO)
    Devi allinearti rigorosamente al "Report di Fact-Checking" fornito:
    - CONTRADDIZIONE: È assolutamente vietato inserire nel testo queste informazioni. Ignorale.
    - NEUTRO: Tratta con cautela, usando un tono dubitativo ("si dice che...", "le leggende narrano...").
    - ENTAILMENT: Usa queste informazioni con assoluta sicurezza.

    ### 6. STILE DELLE CITAZIONI E IMPAGINAZIONE VISIVA
    - NASCONDI I LINK: È ASSOLUTAMENTE VIETATO usare il formato testuale crudo [Fonte: URL]. Nascondi gli URL all'interno del testo usando i collegamenti ipertestuali nativi del Markdown (es: Il biglietto per il [Museo della Pace](https://www.hiroshima-pcf.or.jp/) è di 200 yen). Se c'è [Fonte: Nessuna] o [Fonte: Knowledge Graph], scrivi normalmente senza link.
    - IMMAGINI: DEVI distribuire 2 o 3 immagini ALL'INTERNO del testo per spezzare i paragrafi. NON raggrupparle alla fine. Inserisci in mezzo al testo: [IMAGE: descrizione dettagliata del soggetto visivo in INGLESE].
    """

    # 4. Aggiorna il Contesto Dinamico
    dynamic_context = f"""
    ### CONTESTO EDITORIALE
    - **Argomento Principale**: {topic}
    - **Formato Articolo**: {topic_type}
    - **Focus Editoriale (Angle)**: {angle}
    {f"- **Durata Viaggio**: {duration}" if topic_type == 'itinerary' and duration else ""}
    (ATTENZIONE: L'80% dell'intero articolo DEVE esplorare in profondità il focus editoriale).

    ### DOSSIER FATTUALE (LA TUA BIBBIA PER DATE, PREZZI E LINK)
    {research_summary}

    ### REPORT DI FACT-CHECKING (LoRA Model)
    {fact_check_report}

    ### STORICO KNOWLEDGE GRAPH (K-RAG)
    {kg_summary}
    """

    messages = [SystemMessage(content=system_prompt)]
    
    if human_feedback and human_feedback.lower() not in ["", "approve", "ok", "va bene"]:
        user_message = (
            f"{dynamic_context}\n\n"
            f" REVISIONE EDITORIALE OBBLIGATORIA \n"
            f"Il Direttore ha rifiutato la bozza precedente con questo ordine diretto:\n"
            f"\"{human_feedback}\"\n\n"
            f"Riscrivi l'INTERO articolo obbedendo ciecamente a questo feedback, mantenendo tutte le regole di stile, fact-checking e impaginazione."
        )
    else:
        user_message = f"{dynamic_context}\n\nScrivi l'articolo multimodale basandoti esclusivamente sui dati forniti."

    messages.append(HumanMessage(content=user_message))
    
    # ==========================================
    # 4. QUALITY CONTROL LOOP
    # ==========================================
    max_retries = 3
    draft_text = ""
    min_words = 500
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"   [Drafter] Rigenerazione in corso (Tentativo {attempt + 1}/{max_retries})...")
            
        response = creative_llm.invoke(messages)
        draft_text = response.content
        
        validation_errors = []
        word_count = len(draft_text.split())
        
        if word_count < min_words:
            validation_errors.append(f"- L'articolo è troppo corto ({word_count} parole su {min_words} minime richieste). Espandi la narrativa e i dettagli storici/culturali.")
            

        if not re.search(r'^#\s+', draft_text, re.MULTILINE):
            validation_errors.append("- Manca il Titolo Principale (H1). Inserisci all'inizio dell'articolo '# [Titolo]'.")
        if not re.search(r'^##\s+', draft_text, re.MULTILINE):
            validation_errors.append("- Mancano i Sottotitoli (H2). Dividi il muro di testo in paragrafi leggibili usando '## [Nome Sezione]'.")
            
        
        has_logistics_data = "LOGISTICA E SPOSTAMENTI" in research_summary and "Nessuna informazione disponibile" not in research_summary
        if has_logistics_data:
            if not re.search(r'(?i)^##\s+.*(logistica|muoversi|spostamenti|arrivare|itinerario|trasporti|collegamenti).*', draft_text, re.MULTILINE):
                validation_errors.append("- Hai omesso la sezione sulla Logistica! Dato che il Dossier contiene informazioni preziose su treni, distanze e trasporti, DEVI creare un paragrafo H2 dedicato (es. '## Come muoversi').")


        if validation_errors and attempt < (max_retries - 1):
            print(f"   [Drafter]  Allerta Qualità: Rilevati {len(validation_errors)} errori strutturali. Rimando all'IA...")
            error_msg_str = "\n".join(validation_errors)
            
            messages.append(AIMessage(content=draft_text))
            messages.append(HumanMessage(content=f"La bozza che hai appena generato non supera i controlli di qualità editoriali. Correggi RIGOROSAMENTE i seguenti problemi:\n{error_msg_str}\n\nRiscrivi l'articolo espandendolo tramite dettagli concreti, aneddoti storici o consigli pratici del Dossier. DIVIETO ASSOLUTO: Non usare frasi riempitive, non ripetere concetti (come 'bellezza della natura') e non usare strutture banali ('I visitatori possono...'). Mantieni un ritmo narrativo alto e vario."))
        else:
            if validation_errors:
                print(f"   [Drafter]  Raggiunto il limite di tentativi. Procedo comunque. Errori residui: {len(validation_errors)}")
            else:
                print(f"   [Drafter]  Bozza approvata! Struttura perfetta e lunghezza adeguata ({word_count} parole).")
            break

    # ==========================================
    # 5. CLIP 
    # ==========================================
    print("    [Drafter] Analisi dei segnaposti visivi e interrogazione del modello CLIP...")
    used_image_urls = set()
    
    def replace_with_clip(match):
        clip_query = match.group(1).strip()
        img_url = rag_manager.search_image(clip_query, exclude_urls=list(used_image_urls))
        
        if img_url:
            used_image_urls.add(img_url)
            print(f"      -> Trovata e inserita immagine per: '{clip_query}'")
            return f"\n![{clip_query}]({img_url})\n"
        
        print(f"   -> Nessuna immagine pertinente trovata per: '{clip_query}'")
        return ""
        
    final_draft = re.sub(r'\[IMAGE:\s*(.*?)\]', replace_with_clip, draft_text)
    
    
    reasoning.append({
        "agent": "drafter",
        "thought": f"Analisi del dossier completata. Procedo con la stesura creativa su '{topic}', applicando le regole di fact-checking ({fact_check_report[:30]}...) e iniettando immagini via CLIP.",
        "action": "creative_llm.invoke(messages) + CLIP injection",
        "observation": final_draft[:350]
    })
    
    print("    [Drafter] Bozza multimodale completata e impaginata con successo.")
    
    return {
        "current_draft": final_draft, 
        "reasoning_trace": reasoning,
        "human_feedback": ""
    }

# ═══════════════════════════════════════════
# 5. NODO HUMAN-IN-THE-LOOP
# ═══════════════════════════════════════════
def human_in_the_loop_node(state: BloggerState) -> Command[Literal["kg_updater", "drafter", "researcher", "planner"]]:
    """Presenta la bozza all'utente e gestisce il feedback classificandolo con un LLM."""
    print("\n--- [HumanReviewNode] In attesa di approvazione umana ---")
    

    draft = state.get("current_draft", "")
    word_count = len(draft.split())
    print("\n" + "="*50)
    print(f"BOZZA DA REVISIONARE (Lunghezza: {word_count} parole):")
    print("="*50)
    
    preview_length = 3000
    preview_text = draft[:preview_length]
    if len(draft) > preview_length:
        preview_text += "\n\n... [Il resto dell'articolo è stato troncato per comodità di lettura] ..."
    print(preview_text)
    print("="*50 + "\n")

   
    feedback_payload = interrupt("Cosa ne pensi dell'articolo? (es. 'Perfetto', 'Aggiungi più info sui treni', 'Falla più corta', 'Cambiamo meta, andiamo a Osaka')")
    feedback = str(feedback_payload).strip()
    if not feedback:
        feedback = "approve"
    
    plan = state.get("editorial_plan", [])
    topic = plan[0]["topic"] if plan else state.get("user_input", "Giappone")
    
    reasoning = state.get("reasoning_trace", [])

    
    system_prompt = (
        "Sei un sistema di routing editoriale che classifica il feedback del Direttore su una Guida Turistica.\n"
        f"Il topic (la meta) dell'articolo appena prodotto è '{topic}'.\n"
        "Scegli UNA tra queste decisioni:\n"
        "- 'need_research': L'utente vuole più dettagli, ma SULLA STESSA META dell'articolo (es. 'parlami più dei ristoranti', 'manca la storia del tempio', 'approfondiamo il trekking'; senza specificare una nuova città/meta).\n"
        "- 'change_topic': L'utente vuole scartare questa meta e scrivere una guida su UN ALTRO LUOGO (es. 'cambiamo città', 'andiamo a Kyoto', 'scriviamo di Osaka con focus sul cibo'; quindi sta cambiando topic radicalmente).\n"
        "- 'rewrite': L'utente vuole solo correzioni di stile, tono o lunghezza (es. 'falla più corta', 'usa un tono più romantico', 'togli i bullet points') SULLA STESSA META.\n"
        "- 'approve': L'utente fa complimenti o dà l'ok senza chiedere modifiche specifiche (es. 'ottimo lavoro', 'va benissimo', 'perfetto', 'approvato').\n"

        "\nREGOLE DI PRIORITÀ:\n"
        "1. Se il feedback menziona una città, un quartiere o un'attrazione radicalmente diversa dal topic corrente, scegli SEMPRE 'change_topic'.\n"
        "2. Anche se l'utente specifica focus o approfondimenti del nuovo luogo, SE il topic è diverso allora la decisione resta 'change_topic'.\n"
        "3. 'need_research' può essere scelto SOLO se il feedback riguarda la meta corrente ma richiede fatti o dati nuovi.\n"
        "4. 'rewrite' può essere scelto SOLO se il feedback riguarda la manipolazione del testo già scritto (senza dover cercare nuovi fatti).\n"
        "5. 'approve' si usa solo per il semaforo verde finale.\n"
        "4. DATI PRATICI OBBLIGATORI (CRITICO): se lo ritieni opportuno ed è inerente al discorso DEVE esserci almeno un dato pratico o numerico specifico prelevato dal dossier. Ad esempio: inserisci un prezzo esatto, un orario di apertura, la durata di un tragitto, una tariffa di un hotel o un consiglio su quando evitare la folla. È severamente vietato fare elenchi puntati asettici: devi integrare fluidamente questi numeri all'interno della tua narrazione discorsiva."
    )

    user_prompt = f"FEEDBACK DEL DIRETTORE: {feedback}"

    routing_llm = llm.with_structured_output(FeedbackRouting)
    try:
        decision = routing_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
    except Exception as e:
        print(f"    [Warning] Errore LLM nel routing: {e}. Fallback su 'rewrite'.")
        decision = FeedbackRouting(decision="rewrite", reasoning=f"Fallback per errore LLM: {e}")

    print(f"    [HumanReview] Feedback analizzato: '{feedback[:50]}...'")
    print(f"    [HumanReview] Decisione presa dall'IA: {decision.decision.upper()} (Motivo: {decision.reasoning})")

    dest_map = {
        "need_research": "researcher",
        "change_topic": "planner",
        "rewrite": "drafter",
        "approve": "kg_updater"
    }

    if dest_map.get(decision.decision) == 'change_topic':
        kg_manager.save_active_plan([])

    goto_node = dest_map.get(decision.decision, "drafter")

    reasoning.append({
        "agent": "human_review",
        "thought": f"L'utente ha fornito il seguente feedback: '{feedback}'. L'LLM di routing ha classificato questa richiesta come '{decision.decision}' perché: {decision.reasoning}.",
        "action": f"RouteTo({goto_node})",
        "observation": f"Il flusso viene reindirizzato al nodo '{goto_node}' applicando la logica di memoria corretta."
    })

    update_data = {
        "human_feedback": feedback,
        "reasoning_trace": reasoning 
    }


    delete_msgs = [RemoveMessage(id=m.id) for m in state.get("messages", []) if getattr(m, 'id', None)]
    base_reset = {
        "messages": delete_msgs
    }

    if goto_node == "planner":
        update_data = {
            "user_input": feedback,
            "human_feedback": "",
            "action_results": {},
            "research_summary": "",
            "current_draft": "",
            "editorial_plan": [],
            "kg_summary": "",
            "reasoning_trace": [],
            "fact_check_report": "",
            "tool_call_count": 0,
            **base_reset
        }
    elif goto_node == "researcher":
        update_data = {
            "human_feedback": feedback, 
            "tool_call_count": 0,
            **base_reset
        }
    elif goto_node == "drafter":
        update_data = {
            "human_feedback": feedback, 
            **base_reset
        }

    return Command(update=update_data, goto=goto_node)

# ═══════════════════════════════════════════
# 6. NODO KG UPDATER
# ═══════════════════════════════════════════
def kg_updater_node(state: BloggerState):
    """
    Aggiorna Neo4j estraendo dinamicamente le entità dall'articolo,
    salva il testo completo nel database vettoriale RAG locale,
    aggiorna la coda editoriale e termina l'esecuzione.
    """
    print("\n--- [KG UPDATER] Salvataggio nel Database Travel e RAG ---")
    
    plan = state.get("editorial_plan", [])
    current_draft = state.get("current_draft", "")
    research_summary = state.get("research_summary", "")
    
    reasoning = state.get("reasoning_trace", [])
    
    if not plan:
        print("    [KG UPDATER] Nessun piano trovato, salto l'aggiornamento.")
        return Command(update={"reasoning_trace": reasoning}, goto="__end__")
        
    current_topic = plan[0]["topic"]
    topic_type = plan[0].get("type", "guide")
    angle = plan[0].get("angle", "Generale")
    
    match_titolo = re.search(r'^#\s+(.+)', current_draft, re.MULTILINE)
    vero_titolo_articolo = match_titolo.group(1).strip() if match_titolo else f"{current_topic} - {angle}"

    post_data = {
        "title": vero_titolo_articolo,
        "type": topic_type,
        "topic": current_topic,
        "angle": angle
    }
    

    if not research_summary.strip() and not current_draft.strip():
        print("    [KG UPDATER] ⚠️ Nessun dossier o bozza disponibile. Generazione metadati minimi per evitare errori LLM.")
        extracted_entities = {
            "locations": [current_topic],
            "cities": [], # NUOVO CAMPO
            "prefectures": [],
            "category": [topic_type],
            "local_foods": [],
            "crafts": [],
            "sources": [],
            "claims": []
        }
    else:
        extractor_llm = llm.with_structured_output(TravelMetadataExtractor)
        extraction_prompt = f"""
            Sei un Data Engineer. Analizza il seguente Dossier Turistico e la Bozza dell'articolo.
            Estrai tutti i metadati strutturati richiesti.

            REGOLA CRITICA PER LA GERARCHIA GEOGRAFICA ('mapped_locations'):
            - Per ogni luogo menzionato, crea un oggetto che colleghi in modo inequivocabile l'attrazione (spot), alla sua città (city) e alla sua prefettura (prefecture).
            - Se l'articolo parla in generale di una città senza attrazioni specifiche, compila 'city' e 'prefecture', ma lascia 'spot' vuoto.
            IGNORA categoricamente i luoghi menzionati per puro paragone o come riferimenti ad articoli passati.

            - mapped_locations (LISTA di relazioni Spot-Città-Prefettura)
            - category (LISTA di categorie)
            - local_foods (LISTA di cibi)
            - crafts (LISTA di artigianato)
            - sources (URL delle fonti)
            - claims (fatti chiave)

            DOSSIER DI RICERCA:
            {research_summary[:3000]}

            BOZZA DELL'ARTICOLO:
            {current_draft[:3000]}
            """
        
        try:
            extracted_data = extractor_llm.invoke([
                SystemMessage(content="Estrai le entità dal testo per popolare il Knowledge Graph rispettando rigorosamente la gerarchia geografica."),
                HumanMessage(content=extraction_prompt)
            ])
            extracted_entities = extracted_data.model_dump()
        except Exception as e:
            print(f"[KG UPDATER] Errore durante l'estrazione LLM: {e}. Uso metadati di fallback.")
            extracted_entities = {
                "mapped_locations": [{"spot": current_topic, "city": "", "prefecture": ""}], "category": [topic_type],
                "local_foods": [], "crafts": [], "sources": [], "claims": []
            }
            
    print(f"    [KG UPDATER] Entità geografiche estratte: {len(extracted_entities.get('mapped_locations', []))} location mappate.")
    # ==========================================
    # SAVATAGGIO NEL KNOWLEDGE GRAPH E NEL RAG
    # ==========================================
    try:
        
        risultato = kg_manager.update_after_approval(post_data, extracted_entities)
        print(f"    [KG UPDATER]  Post e relazioni salvati su Neo4j: {risultato.get('saved_post')}")
        
        
        print(f"[KG UPDATER] Salvataggio del testo completo nel RAG locale...")
        lista_prefetture = extracted_entities.get("prefectures", [])
        prefettura_principale = lista_prefetture[0] if lista_prefetture else "Sconosciuta"
        doc = Document(
            page_content=f"ARTICOLO BLOG PRECEDENTE:\nTitolo: {vero_titolo_articolo}\nTipo: {topic_type}\nFocus: {angle}\n\nTesto completo:\n{current_draft}",
            metadata={
                "source": "blog_interno",
                "name": vero_titolo_articolo,  
                "topic": current_topic,        
                "type": topic_type,
                "prefecture": prefettura_principale,
                "date": datetime.now().isoformat() 
            }
        )
        
        rag_manager.add_documents([doc])
        print("[KG UPDATER]  Articolo inserito con successo in ChromaDB.")

        reasoning.append({
            "agent": "kg_updater",
            "thought": f"L'articolo '{current_topic}' è stato approvato dall'utente. Devo estrarre le entità e salvarlo sia nel Knowledge Graph (Neo4j) che nel Vector DB (Chroma).",
            "action": "kg_manager.update_after_approval() & rag_manager.add_documents()",
            "observation": f"Post '{current_topic}' salvato con successo. Entità collegate: {extracted_entities.get('locations')}."
        })
        
    except Exception as e:
        print(f"[KG UPDATER] Errore durante l'estrazione LLM o il salvataggio: {e}")
        reasoning.append({
            "agent": "kg_updater",
            "thought": "Tentativo di salvare i dati nel KG e RAG.",
            "action": "Database update",
            "observation": f"Errore critico durante il salvataggio: {e}"
        })
        
    # ==========================================
    # 5. GESTIONE DELLA CODA EDITORIALE
    # ==========================================
    active_plan = kg_manager.get_active_plan_status()
    
    if active_plan and len(active_plan) > 0:
        try:
            post_pubblicato = active_plan.pop(0)
            print(f"[KG UPDATER] Rimosso '{post_pubblicato['topic']}' dalla coda editoriale.")
            kg_manager.save_active_plan(active_plan)

            next_topic = active_plan[0]['topic'] if len(active_plan) > 0 else "Nessuno (piano completato)"

            reasoning.append({
                "agent": "kg_updater",
                "thought": "Aggiornamento della coda del piano editoriale dopo la pubblicazione.",
                "action": "active_plan.pop(0) & kg_manager.save_active_plan()",
                "observation": f"L'articolo '{post_pubblicato['topic']}' è stato formalmente pubblicato e rimosso dalla coda. Prossimo articolo previsto: {next_topic}."
            })

        except Exception as e:
            print(f"[KG UPDATER] Errore DB nella coda editoriale: {e}")
            return Command(update={"reasoning_trace": reasoning}, goto="__end__")
        
       
        if len(active_plan) > 0:
            print(f"[KG UPDATER] Il prossimo itinerario in coda è: '{next_topic}'. Esecuzione terminata.")
            return Command(
                update={"full_calendar": active_plan, "reasoning_trace": reasoning}, 
                goto="__end__"
            )
        else:
            print(f"[KG UPDATER] Calendario completato! Tutti gli articoli in programma sono stati pubblicati.")
            kg_manager.save_active_plan([])
            return Command(
                update={"editorial_plan": [], "full_calendar": [], "reasoning_trace": reasoning},
                goto="__end__" 
            )

    return Command(update={"reasoning_trace": reasoning}, goto="__end__")
