# Japan Travel AI Blogger 🇯🇵

**Copilot agentico per la gestione di un blog di viaggi in Giappone** — progetto per il corso di *Cognitive Computing & Artificial Intelligence (CCAI) 2025-2026*.

Il sistema è un'architettura multi-agente orchestrata con **LangGraph** che assiste un blogger nell'intero ciclo editoriale: pianificazione del calendario, ricerca delle informazioni, verifica fattuale, stesura della bozza e revisione umana (human-in-the-loop) prima della pubblicazione. Integra un approccio **K-RAG** (Knowledge Graph + RAG), retrieval multimodale con CLIP e un modello di fact-checking specializzato tramite **fine-tuning LoRA**.

---

## Architettura in breve

Il grafo di stato (`blog_agent`) è composto dai seguenti nodi:

`planner` → `researcher` (sotto-grafo ReAct) → `recap` → `fact_checker` → `drafter` → `human_review` → `kg_updater`

- **Planner** — interroga il Knowledge Graph per gap di copertura e festival stagionali, e genera un calendario editoriale.
- **Researcher** — sotto-grafo ReAct che alterna ricerca web (Tavily), RAG e interrogazione del KG.
- **Recap** — distilla i dati grezzi in un dossier Markdown strutturato con Citation Guard anti-allucinazione.
- **Fact Checker** — estrae i claim e li valida con il modello LoRA (mDeBERTa).
- **Drafter** — scrive l'articolo con Quality Control Loop e impaginazione multimodale (CLIP).
- **Human Review** — interruzione obbligatoria per l'approvazione/modifica dell'utente.
- **KG Updater** — persiste l'articolo approvato su Neo4j e ChromaDB.

---

## Prerequisiti

Prima di iniziare assicurati di avere:

1. **Python 3.11 – 3.13** (il progetto è stato sviluppato con Python 3.13).
2. **Git** per clonare la repository.
3. Un'istanza **Neo4j** accessibile. Le opzioni consigliate sono:
   - [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier, nessuna installazione locale)
4. Le seguenti **API key** (tutte gratuite nei rispettivi tier base):
   - **Groq** → https://console.groq.com (obbligatoria, è l'LLM principale)
   - **Tavily** → https://tavily.com (obbligatoria, motore di ricerca web)
   - **LangSmith** → https://smith.langchain.com (opzionale, solo per l'osservabilità/tracing)

> ⚠️ Al primo avvio verranno scaricati automaticamente da HuggingFace i modelli `all-MiniLM-L6-v2` (embedding testuali) e `openai/clip-vit-base-patch32` (embedding immagini). Serve quindi una connessione a internet e qualche GB di spazio disco.

---

## Struttura del progetto

Dopo il clone, la cartella di lavoro effettiva è `TravelBlogger/`:

```
Travel_Blogger_CCAI_Project/
└── TravelBlogger/                 ← esegui tutti i comandi da qui
    ├── langgraph.json             # configurazione LangGraph (grafo + .env)
    ├── requirements.txt           # dipendenze Python
    ├── example_env.txt            # template delle variabili d'ambiente
    ├── init_kg.py                 # script di inizializzazione del Knowledge Graph
    ├── .env                       # ← DA CREARE tu (vedi sotto)
    ├── model/
    │   ├── fact_checker_lora/     # adapter LoRA del fact-checker 
    │   ├── finetune.ipynb         # notebook di fine-tuning
    │   └── generate_factcheck_dataset.py
    └── src/
        ├── graph.py               # definizione del grafo 
        ├── state.py               # BloggerState 
        ├── schemas.py             # schemi Pydantic
        ├── kg_manager.py          # gestione Neo4j
        ├── rag_manager.py         # gestione ChromaDB + CLIP + BM25
        ├── agents/nodes.py        # logica di tutti i nodi
        └── tools/tools.py         # i 5 tool dell'agente
```

Al primo avvio verrà creata automaticamente anche la cartella `chroma_db_travel/` .

---

## Installazione

Da terminale:

```bash
# 1. Clona la repository
git clone https://github.com/<tuo-utente>/Travel_Blogger_CCAI_Project.git
cd Travel_Blogger_CCAI_Project/TravelBlogger

# 2. Crea e attiva un ambiente virtuale
python -m venv venv

# su macOS / Linux
source venv/bin/activate
# su Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3. Installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configurazione del file `.env`

Nella cartella `TravelBlogger/` crea un file chiamato **`.env`** (puoi partire copiando `example_env.txt`):

```bash
cp example_env.txt .env
```

Poi apri `.env` e compila i valori. Le variabili richieste sono:

| Variabile | Obbligatoria | Descrizione |
|---|:---:|---|
| `GROQ_API_KEY` | ✅ | Chiave API di Groq (LLM principale `llama-3.3-70b-versatile`). |
| `TAVILY_API_KEY` | ✅ | Chiave API di Tavily per la ricerca web. |
| `NEO4J_URI` | ✅ | URI dell'istanza Neo4j (es. `neo4j+s://xxxx.databases.neo4j.io` per Aura, oppure `bolt://localhost:7687` in locale). |
| `NEO4J_USERNAME` | ✅ | Username Neo4j (di default `neo4j`). |
| `NEO4J_PASSWORD` | ✅ | Password Neo4j. |
| `LANGCHAIN_TRACING_V2` | ⬜ | `"true"` per abilitare il tracing su LangSmith, altrimenti `"false"`. |
| `LANGCHAIN_ENDPOINT` | ⬜ | `"https://api.smith.langchain.com"`. |
| `LANGCHAIN_API_KEY` | ⬜ | Chiave API di LangSmith (solo se il tracing è attivo). |
| `LANGCHAIN_PROJECT` | ⬜ | Nome del progetto su LangSmith (es. `"TravelBlogger_CCAI"`). |

Esempio di `.env` compilato:

```env
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxx"

NEO4J_URI="neo4j+s://abcd1234.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="la-tua-password"

LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_xxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_PROJECT="TravelBlogger_CCAI"
```

---

## Setup del Knowledge Graph (Neo4j)

Una volta che l'istanza Neo4j è attiva e le credenziali sono nel `.env`, devi **inizializzare l'ontologia** del grafo. Questo passaggio va eseguito **una sola volta** (o ogni volta che vuoi ripartire da un grafo pulito) e crea i constraint di unicità e i nodi di base (le 8 regioni e le 47 prefetture del Giappone):

```bash
python init_kg.py
```

Se tutto è configurato correttamente vedrai messaggi come:

```
Inizializzazione Ontologia Knowledge Graph - Guida Turistica Giappone
Constraints di unicità applicati con successo.
Inserite 8 regioni.
...
```

Il grafo viene poi arricchito **incrementalmente e in automatico** dal nodo `kg_updater` ogni volta che approvi un articolo.

---

## Avvio del sistema

Il progetto usa **LangGraph Studio** tramite la CLI di LangGraph. Dalla cartella `TravelBlogger/` (con l'ambiente virtuale attivo) esegui:

```bash
langgraph dev
```

Il comando avvia un server locale e apre (o ti fornisce il link a) l'interfaccia **LangGraph Studio**, dalla quale puoi interagire con il grafo `blog_agent` definito in `langgraph.json`.

Nell'interfaccia troverai:
- un campo di **input** dove inserire la richiesta (`user_input`);
- la visualizzazione **live** del grafo e dello stato a ogni nodo;
- i punti di **interrupt** (revisione umana) dove il flusso si ferma in attesa del tuo feedback.

---


## Come si usa

Un ciclo tipico di generazione di un articolo:

1. **Inserisci una richiesta** nel campo di input, ad esempio:
   - `Scrivimi degli articoli su Kyoto` → genera un calendario di 3 guide con angoli editoriali diversi;
   - `Scrivimi un articolo itinerario tra Osaka e Nara` → genera 1 solo post di tipo itinerario.
2. **Approva il calendario** — il Planner propone le mete e le motiva; puoi approvare o chiedere modifiche in linguaggio naturale.
3. **Attendi ricerca, recap e fact-checking** — l'agente lavora in autonomia (è il passaggio più lungo, ~1-2 minuti).
4. **Rivedi la bozza** (human-in-the-loop) — al nodo `human_review` il flusso si ferma e ti mostra l'articolo. Puoi rispondere con:
   - un'**approvazione** (es. *"perfetto"*, *"va bene"*) → l'articolo viene salvato nel KG;
   - una richiesta di **modifica stilistica** (es. *"falla più corta"*) → torna al Drafter;
   - una richiesta di **approfondimento** (es. *"aggiungi più info sui treni"*) → torna al Researcher;
   - un **cambio di meta** (es. *"cambiamo città, andiamo a Tokyo"*) → riparte dal Planner.
5. **Salvataggio** — solo dopo l'approvazione, il `kg_updater` aggiorna Neo4j e ChromaDB.

> ℹ️ Il Knowledge Graph viene aggiornato **esclusivamente dopo l'approvazione umana**: nulla viene persistito senza il tuo via libera.

---

## Il modello fine-tuned (fact-checker)

Il fact-checker è un modello **mDeBERTa-v3-base-mnli-xnli** adattato al dominio turistico giapponese tramite **LoRA**. L'adapter addestrato è già incluso in `model/fact_checker_lora/`, quindi **non è necessario ri-addestrare nulla** per usare il sistema.

Se vuoi riprodurre il fine-tuning:
- il notebook completo è in `model/finetune.ipynb` (pensato per un ambiente con GPU, es. Kaggle/Colab);
- il dataset NLI in italiano è in `model/factcheck_dataset_japan.csv` ed è generabile con `model/generate_factcheck_dataset.py`.

---

*Progetto realizzato da Gabriele Florio ed Enricomaria Di Rosolini — CCAI 2025-2026.*
