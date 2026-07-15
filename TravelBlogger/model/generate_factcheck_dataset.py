# -*- coding: utf-8 -*-
"""
Generatore del dataset di Fact-Checking (NLI) per il dominio "Viaggi in Giappone".
==================================================================================

Produce un CSV con colonne:
    fact_id  : identificatore del fatto sorgente (usato per lo split a gruppi -> no leakage)
    context  : paragrafo "fonte di verità" (come quelli estratti dal RAG/KG dell'agente)
    claim    : affermazione da verificare (come quelle prodotte dal Drafter)
    label    : 0 = CONTRADICTION (il contesto smentisce il claim)
               1 = ENTAILMENT    (il contesto conferma il claim)
               2 = NEUTRAL       (il contesto non contiene informazioni sufficienti)

Strategia (dichiarata nella relazione, punto 4 delle linee guida):
- Dataset SINTETICO ma controllato: partiamo da una base di ~60 fatti reali sul
  turismo in Giappone (prezzi, orari, trasporti, etichetta, stagioni, logistica).
- ENTAILMENT: parafrasi templatiche del fatto.
- CONTRADICTION: perturbazioni controllate (numeri alterati, antonimi
  gratuito/a pagamento, consentito/vietato, negazioni).
- NEUTRAL: due tipologie, in ordine di difficoltà:
    (a) claim su un ATTRIBUTO NON DICHIARATO della stessa entità (hard negatives)
    (b) claim su un'entità diversa (easy negatives)
- Lo split train/val/test viene fatto NEL NOTEBOOK con GroupShuffleSplit su
  fact_id: tutte le varianti di uno stesso fatto finiscono nello stesso split.

Uso:
    python generate_factcheck_dataset.py --out factcheck_dataset_japan.csv --seed 42
"""

import argparse
import csv
import random

# ----------------------------------------------------------------------------
# 1. BASE DI FATTI
# Ogni fatto: id, entità, categoria, contesti (2 parafrasi), claim veri,
# claim falsi, claim neutri "hard" (stessa entità, attributo non dichiarato).
# ----------------------------------------------------------------------------

FACTS = [
    # ------------------------- TEMPLI E ATTRAZIONI -------------------------
    dict(
        id="kinkakuji",
        contexts=[
            "Il Kinkaku-ji, il celebre Padiglione d'Oro di Kyoto, è aperto tutti i giorni dalle 9:00 alle 17:00. Il biglietto d'ingresso costa 500 yen per gli adulti e 300 yen per gli studenti delle scuole elementari e medie.",
            "Il Padiglione d'Oro (Kinkaku-ji) di Kyoto accoglie i visitatori ogni giorno dalle 9:00 alle 17:00; l'ingresso per gli adulti costa 500 yen, mentre per gli studenti di elementari e medie il prezzo è di 300 yen.",
        ],
        entail=[
            "L'ingresso al Kinkaku-ji costa 500 yen per un adulto.",
            "Il Padiglione d'Oro chiude alle 17:00.",
            "Il Kinkaku-ji è visitabile ogni giorno della settimana.",
            "Gli studenti delle scuole medie pagano 300 yen per entrare al Kinkaku-ji.",
        ],
        contradict=[
            "L'ingresso al Padiglione d'Oro di Kyoto è gratuito.",
            "Il Kinkaku-ji chiude alle 21:00 per le visite serali.",
            "Il biglietto per il Kinkaku-ji costa 1500 yen per gli adulti.",
            "Il Kinkaku-ji è chiuso il lunedì.",
        ],
        neutral=[
            "Al Kinkaku-ji è possibile acquistare amuleti omamori vicino all'uscita.",
            "Il Kinkaku-ji è raggiungibile in autobus dalla stazione di Kyoto in circa 40 minuti.",
            "Il giardino del Kinkaku-ji ospita una casa da tè storica.",
        ],
    ),
    dict(
        id="fushimi_inari",
        contexts=[
            "Il santuario Fushimi Inari Taisha di Kyoto è famoso per i suoi migliaia di torii rossi. L'accesso è gratuito e il santuario è aperto 24 ore su 24, tutti i giorni dell'anno.",
            "Fushimi Inari Taisha, il santuario di Kyoto noto per i tunnel di torii vermigli, non richiede alcun biglietto d'ingresso ed è visitabile a qualsiasi ora, ogni giorno dell'anno.",
        ],
        entail=[
            "Visitare il Fushimi Inari non costa nulla.",
            "Il Fushimi Inari Taisha è aperto anche di notte.",
            "Non serve un biglietto per entrare al Fushimi Inari.",
            "Il santuario Fushimi Inari è celebre per i suoi torii rossi.",
        ],
        contradict=[
            "L'ingresso al Fushimi Inari costa 400 yen.",
            "Il Fushimi Inari chiude ogni sera alle 18:00.",
            "Il santuario Fushimi Inari è chiuso durante le festività di Capodanno.",
            "Per salire al Fushimi Inari è obbligatorio prenotare una fascia oraria.",
        ],
        neutral=[
            "La salita completa al monte Inari richiede circa due o tre ore.",
            "Lungo il percorso del Fushimi Inari si trovano bancarelle di street food.",
            "Il Fushimi Inari è dedicato alla divinità del riso e del commercio.",
        ],
    ),
    dict(
        id="sensoji",
        contexts=[
            "Il Senso-ji, nel quartiere di Asakusa a Tokyo, è il tempio più antico della città, fondato nel 645. L'ingresso al tempio è gratuito; la sala principale è aperta dalle 6:00 alle 17:00 (dalle 6:30 da ottobre a marzo).",
            "Fondato nel 645, il Senso-ji di Asakusa è il tempio più antico di Tokyo. Non si paga alcun biglietto e la sala principale apre alle 6:00 (6:30 nei mesi da ottobre a marzo) e chiude alle 17:00.",
        ],
        entail=[
            "Il Senso-ji è il tempio più antico di Tokyo.",
            "Entrare al Senso-ji è gratuito.",
            "Il Senso-ji si trova nel quartiere di Asakusa.",
            "In inverno la sala principale del Senso-ji apre alle 6:30.",
        ],
        contradict=[
            "Il Senso-ji è stato fondato nel 1945.",
            "Per entrare al Senso-ji si paga un biglietto di 300 yen.",
            "Il Senso-ji si trova nel quartiere di Shibuya.",
            "La sala principale del Senso-ji resta aperta fino a mezzanotte.",
        ],
        neutral=[
            "Davanti al Senso-ji si estende la via commerciale Nakamise-dori.",
            "Al Senso-ji è possibile estrarre gli omikuji, i biglietti della fortuna.",
            "Il grande lampione rosso del Senso-ji si trova alla porta Kaminarimon.",
        ],
    ),
    dict(
        id="teamlab",
        contexts=[
            "Il museo digitale teamLab Planets di Tokyo richiede l'acquisto del biglietto online con data e ora prestabilite. Il prezzo per gli adulti è di 3800 yen e si visita a piedi nudi, poiché alcune installazioni prevedono di camminare nell'acqua.",
            "Per visitare teamLab Planets a Tokyo bisogna comprare online un biglietto con fascia oraria; un adulto paga 3800 yen. Il percorso si affronta scalzi perché in alcune sale si cammina nell'acqua.",
        ],
        entail=[
            "Il biglietto per teamLab Planets si acquista online con fascia oraria.",
            "Un adulto paga 3800 yen per teamLab Planets.",
            "A teamLab Planets si cammina a piedi nudi.",
            "Alcune installazioni di teamLab Planets prevedono di camminare nell'acqua.",
        ],
        contradict=[
            "A teamLab Planets si può entrare liberamente senza prenotazione.",
            "Il biglietto per teamLab Planets costa 1000 yen per gli adulti.",
            "A teamLab Planets è obbligatorio tenere le scarpe per tutto il percorso.",
            "L'ingresso a teamLab Planets è gratuito la prima domenica del mese.",
        ],
        neutral=[
            "teamLab Planets si trova nella zona di Toyosu.",
            "La visita a teamLab Planets dura in media circa due ore.",
            "Vicino a teamLab Planets si trova il mercato del pesce di Toyosu.",
        ],
    ),
    dict(
        id="shibuya_sky",
        contexts=[
            "L'osservatorio Shibuya Sky si trova al 47° piano del grattacielo Shibuya Scramble Square, a 229 metri di altezza. È aperto dalle 10:00 alle 22:30, con ultimo ingresso alle 21:20.",
            "Shibuya Sky è la terrazza panoramica in cima allo Shibuya Scramble Square, al 47° piano, 229 metri sopra la città. Gli orari di apertura vanno dalle 10:00 alle 22:30 e l'ultimo ingresso è consentito alle 21:20.",
        ],
        entail=[
            "Shibuya Sky si trova a 229 metri di altezza.",
            "L'ultimo ingresso a Shibuya Sky è alle 21:20.",
            "Shibuya Sky è aperto fino alle 22:30.",
            "L'osservatorio Shibuya Sky è al 47° piano dello Shibuya Scramble Square.",
        ],
        contradict=[
            "Shibuya Sky chiude alle 18:00.",
            "L'osservatorio Shibuya Sky si trova al 10° piano.",
            "Shibuya Sky si trova a 634 metri di altezza.",
            "A Shibuya Sky si può entrare fino a cinque minuti prima della chiusura.",
        ],
        neutral=[
            "Da Shibuya Sky nelle giornate limpide si può vedere il monte Fuji.",
            "A Shibuya Sky non è consentito portare treppiedi fotografici.",
            "I biglietti per Shibuya Sky al tramonto si esauriscono con giorni di anticipo.",
        ],
    ),
    dict(
        id="himeji",
        contexts=[
            "Il castello di Himeji, patrimonio UNESCO, è aperto dalle 9:00 alle 17:00 (ultimo ingresso alle 16:00). Il biglietto costa 1000 yen per gli adulti ed è chiuso solo il 29 e 30 dicembre.",
            "Patrimonio dell'umanità UNESCO, il castello di Himeji si visita dalle 9:00 alle 17:00 con ultimo accesso alle 16:00; gli adulti pagano 1000 yen e il castello chiude soltanto il 29 e il 30 dicembre.",
        ],
        entail=[
            "Il castello di Himeji è patrimonio UNESCO.",
            "L'ingresso al castello di Himeji costa 1000 yen per gli adulti.",
            "L'ultimo ingresso al castello di Himeji è alle 16:00.",
            "Il castello di Himeji chiude solo due giorni all'anno, a fine dicembre.",
        ],
        contradict=[
            "Il castello di Himeji è chiuso ogni lunedì.",
            "L'ingresso al castello di Himeji è gratuito.",
            "Si può entrare al castello di Himeji fino alle 20:00.",
            "Il biglietto del castello di Himeji costa 100 yen.",
        ],
        neutral=[
            "Il castello di Himeji è soprannominato 'castello dell'airone bianco'.",
            "Da Osaka si raggiunge Himeji in circa un'ora di treno.",
            "In primavera il parco del castello di Himeji si riempie di ciliegi in fiore.",
        ],
    ),
    dict(
        id="ghibli_museum",
        contexts=[
            "Il Museo Ghibli di Mitaka, a Tokyo, vende i biglietti solo su prenotazione anticipata: non è possibile acquistarli in loco. All'interno del museo è vietato fotografare e filmare.",
            "Per il Museo Ghibli di Mitaka i biglietti vanno prenotati in anticipo, non esiste biglietteria in loco. Nelle sale interne fotografie e video sono vietati.",
        ],
        entail=[
            "Non si possono comprare i biglietti del Museo Ghibli direttamente in loco.",
            "Al Museo Ghibli è vietato scattare fotografie all'interno.",
            "I biglietti del Museo Ghibli richiedono la prenotazione anticipata.",
            "Filmare all'interno del Museo Ghibli non è consentito.",
        ],
        contradict=[
            "I biglietti del Museo Ghibli si comprano comodamente alla cassa il giorno stesso.",
            "All'interno del Museo Ghibli si può fotografare liberamente.",
            "Il Museo Ghibli non richiede alcuna prenotazione.",
            "Nel Museo Ghibli è permesso girare video nelle sale espositive.",
        ],
        neutral=[
            "Il Museo Ghibli espone un gigantesco robot del film Laputa sul tetto.",
            "Il Museo Ghibli si raggiunge con una passeggiata dal parco Inokashira.",
            "Il cinema interno del Museo Ghibli proietta cortometraggi esclusivi.",
        ],
    ),
    # ------------------------------ TRASPORTI ------------------------------
    dict(
        id="jrpass",
        contexts=[
            "Il Japan Rail Pass ordinario da 7 giorni costa 50000 yen e consente viaggi illimitati sulla rete JR, inclusi gli Shinkansen Hikari e Kodama, ma non è valido sui treni Nozomi e Mizuho.",
            "Con 50000 yen si acquista il Japan Rail Pass ordinario di 7 giorni, che permette di viaggiare senza limiti sui treni JR e sugli Shinkansen Hikari e Kodama; restano esclusi i Nozomi e i Mizuho.",
        ],
        entail=[
            "Il Japan Rail Pass da 7 giorni costa 50000 yen.",
            "Il Japan Rail Pass non è valido sui treni Nozomi.",
            "Con il Japan Rail Pass si può salire sugli Shinkansen Hikari.",
            "I treni Mizuho sono esclusi dal Japan Rail Pass.",
        ],
        contradict=[
            "Il Japan Rail Pass da 7 giorni costa 29650 yen.",
            "Il Japan Rail Pass copre anche i treni Nozomi senza supplementi.",
            "Con il Japan Rail Pass non si può viaggiare su nessuno Shinkansen.",
            "Il Japan Rail Pass di una settimana costa 10000 yen.",
        ],
        neutral=[
            "Il Japan Rail Pass può essere acquistato anche online sul sito ufficiale.",
            "Esiste una versione Green del Japan Rail Pass per la prima classe.",
            "Il Japan Rail Pass conviene soprattutto per itinerari con molte tratte lunghe.",
        ],
    ),
    dict(
        id="shinkansen_tokyo_kyoto",
        contexts=[
            "Lo Shinkansen Nozomi collega Tokyo a Kyoto in circa 2 ore e 15 minuti. Un biglietto di sola andata con posto riservato costa circa 14000 yen.",
            "Il treno ad alta velocità Nozomi impiega circa 2 ore e 15 minuti da Tokyo a Kyoto; la sola andata con posto riservato costa intorno ai 14000 yen.",
        ],
        entail=[
            "Da Tokyo a Kyoto con il Nozomi ci vogliono circa 2 ore e un quarto.",
            "Un biglietto Tokyo-Kyoto sul Nozomi con posto riservato costa circa 14000 yen.",
            "Il Nozomi è un treno che collega Tokyo e Kyoto.",
        ],
        contradict=[
            "Il viaggio in Shinkansen da Tokyo a Kyoto dura circa 6 ore.",
            "Il biglietto del Nozomi da Tokyo a Kyoto costa circa 3000 yen.",
            "Il Nozomi impiega 45 minuti da Tokyo a Kyoto.",
        ],
        neutral=[
            "Sedendosi sul lato destro del treno verso Kyoto si può vedere il monte Fuji.",
            "Sui Nozomi sono disponibili vagoni con posti non riservati.",
            "Alla stazione di Tokyo si possono comprare bento da gustare a bordo.",
        ],
    ),
    dict(
        id="suica",
        contexts=[
            "Le carte ricaricabili Suica e Pasmo si usano su treni, metropolitane e autobus di Tokyo e in gran parte del Giappone, e permettono anche di pagare nei konbini e ai distributori automatici.",
            "Suica e Pasmo sono carte prepagate valide su metro, treni e bus a Tokyo e in molte regioni del Giappone; si possono usare anche per pagare nei minimarket e ai distributori automatici.",
        ],
        entail=[
            "Con la Suica si può pagare nei konbini.",
            "Le carte Pasmo funzionano anche sugli autobus di Tokyo.",
            "Suica e Pasmo sono carte ricaricabili.",
            "La Suica si può usare ai distributori automatici.",
        ],
        contradict=[
            "La Suica è valida solo sulla metropolitana e non nei negozi.",
            "Le carte Pasmo non funzionano fuori da Tokyo.",
            "La Suica è un abbonamento mensile non ricaricabile.",
        ],
        neutral=[
            "La Suica può essere aggiunta al wallet dello smartphone.",
            "Per la Suica fisica è previsto un deposito cauzionale di 500 yen.",
            "Esiste una versione Welcome Suica pensata per i turisti.",
        ],
    ),
    dict(
        id="narita_express",
        contexts=[
            "Il Narita Express (N'EX) collega l'aeroporto di Narita alla stazione di Tokyo in circa 60 minuti; il biglietto di sola andata costa circa 3070 yen e tutti i posti sono a prenotazione obbligatoria.",
            "Dall'aeroporto di Narita alla stazione di Tokyo, il Narita Express impiega circa un'ora e costa circa 3070 yen a tratta; la prenotazione del posto è obbligatoria per tutti i passeggeri.",
        ],
        entail=[
            "Il Narita Express impiega circa un'ora per arrivare alla stazione di Tokyo.",
            "Sul Narita Express la prenotazione del posto è obbligatoria.",
            "Un biglietto del Narita Express costa circa 3070 yen.",
        ],
        contradict=[
            "Il Narita Express arriva alla stazione di Tokyo in 15 minuti.",
            "Sul Narita Express ci si siede liberamente senza prenotazione.",
            "Il Narita Express costa circa 300 yen a tratta.",
        ],
        neutral=[
            "In alternativa al Narita Express esiste il Keisei Skyliner per Ueno.",
            "Il Narita Express dispone di spazi dedicati ai bagagli con lucchetto.",
            "Con alcuni pass turistici il Narita Express è incluso nel prezzo.",
        ],
    ),
    dict(
        id="taxi_japan",
        contexts=[
            "Nei taxi giapponesi le portiere posteriori si aprono e si chiudono automaticamente, comandate dall'autista. Non è consuetudine lasciare la mancia e la tariffa base a Tokyo parte da circa 500 yen.",
            "In Giappone le porte posteriori dei taxi sono automatiche e gestite dal conducente. La mancia non è prevista e a Tokyo la corsa parte da una tariffa base di circa 500 yen.",
        ],
        entail=[
            "Le portiere posteriori dei taxi giapponesi si aprono automaticamente.",
            "In Giappone non si lascia la mancia al tassista.",
            "A Tokyo la tariffa base del taxi parte da circa 500 yen.",
        ],
        contradict=[
            "Nei taxi giapponesi bisogna sempre aprire la portiera da soli.",
            "In Giappone è obbligatorio lasciare una mancia del 15% al tassista.",
            "La tariffa base dei taxi a Tokyo è di circa 5000 yen.",
        ],
        neutral=[
            "Molti taxi giapponesi accettano il pagamento con carte IC.",
            "Di notte le tariffe dei taxi giapponesi prevedono un supplemento.",
            "A Tokyo si possono chiamare i taxi tramite app come GO.",
        ],
    ),
    # ------------------------------- ETICHETTA ------------------------------
    dict(
        id="onsen_tattoo",
        contexts=[
            "In molti onsen tradizionali giapponesi l'ingresso alle persone tatuate è ancora vietato o limitato; alcune strutture consentono l'accesso se il tatuaggio viene coperto con un cerotto. Prima di entrare in vasca è obbligatorio lavarsi accuratamente il corpo.",
            "Numerosi onsen tradizionali in Giappone vietano o limitano l'ingresso a chi ha tatuaggi, anche se alcune strutture li accettano se coperti con un apposito cerotto. È inoltre obbligatorio lavarsi bene il corpo prima di immergersi.",
        ],
        entail=[
            "In molti onsen tradizionali i tatuaggi non sono ammessi.",
            "Prima di entrare nella vasca di un onsen bisogna lavarsi.",
            "Alcuni onsen accettano i tatuaggi se coperti con un cerotto.",
        ],
        contradict=[
            "Tutti gli onsen giapponesi accettano senza problemi le persone tatuate.",
            "Negli onsen ci si immerge in vasca senza bisogno di lavarsi prima.",
            "In Giappone nessun onsen pone limitazioni ai tatuaggi.",
        ],
        neutral=[
            "Molti onsen forniscono asciugamani a noleggio a pagamento.",
            "Alcuni ryokan dispongono di vasche private prenotabili.",
            "L'acqua degli onsen è ricca di minerali con proprietà diverse.",
        ],
    ),
    dict(
        id="tipping",
        contexts=[
            "In Giappone non esiste la cultura della mancia: lasciarla nei ristoranti può risultare scortese o creare imbarazzo. Il conto si paga generalmente alla cassa, non al tavolo.",
            "Lasciare la mancia in Giappone non è consuetudine e nei ristoranti può addirittura mettere in imbarazzo il personale. Di norma si paga alla cassa e non al tavolo.",
        ],
        entail=[
            "In Giappone non si lascia la mancia al ristorante.",
            "Nei ristoranti giapponesi si paga di solito alla cassa.",
            "Lasciare la mancia in Giappone può creare imbarazzo.",
        ],
        contradict=[
            "In Giappone è considerato obbligatorio lasciare almeno il 10% di mancia.",
            "Nei ristoranti giapponesi il pagamento avviene sempre al tavolo.",
            "I camerieri giapponesi si aspettano una mancia generosa.",
        ],
        neutral=[
            "In molti ristoranti giapponesi si ordina tramite distributori di biglietti.",
            "Alcuni izakaya applicano un piccolo coperto chiamato otoshi.",
            "In Giappone molti locali offrono acqua o tè gratuitamente.",
        ],
    ),
    dict(
        id="chopsticks",
        contexts=[
            "Nel galateo giapponese è considerato di cattivo gusto piantare le bacchette verticalmente nel riso, perché richiama i riti funebri; allo stesso modo non si deve passare il cibo da bacchette a bacchette.",
            "Secondo l'etichetta giapponese non bisogna mai infilare le bacchette in verticale nella ciotola di riso, gesto legato ai funerali, né passarsi il cibo direttamente da bacchette a bacchette.",
        ],
        entail=[
            "Piantare le bacchette verticalmente nel riso è considerato maleducato in Giappone.",
            "Passare il cibo da bacchette a bacchette è contrario al galateo giapponese.",
            "Il gesto delle bacchette verticali nel riso richiama i riti funebri.",
        ],
        contradict=[
            "In Giappone infilare le bacchette in verticale nel riso è un segno di rispetto.",
            "Passarsi il cibo di bacchette in bacchette è un'usanza apprezzata in Giappone.",
            "Il galateo giapponese non prevede alcuna regola sull'uso delle bacchette.",
        ],
        neutral=[
            "Nei ristoranti giapponesi le bacchette usa e getta si chiamano waribashi.",
            "In Giappone è normale avvicinare la ciotola alla bocca per mangiare il riso.",
            "Molti locali giapponesi mettono a disposizione anche forchette su richiesta.",
        ],
    ),
    dict(
        id="shoes_off",
        contexts=[
            "In Giappone è obbligatorio togliersi le scarpe prima di entrare nelle case private, nei ryokan e in molti templi; all'ingresso (genkan) si trovano spesso pantofole da indossare all'interno.",
            "Entrando in una casa giapponese, in un ryokan o in molti templi bisogna togliersi le scarpe nel genkan, l'area di ingresso, dove di solito sono disponibili pantofole per l'interno.",
        ],
        entail=[
            "Nelle case giapponesi ci si toglie le scarpe all'ingresso.",
            "In molti templi giapponesi è necessario togliersi le scarpe.",
            "Il genkan è l'area d'ingresso dove si lasciano le scarpe.",
        ],
        contradict=[
            "In Giappone si entra tranquillamente in casa con le scarpe.",
            "Nei ryokan è vietato togliersi le scarpe.",
            "Nei templi giapponesi è obbligatorio tenere le scarpe ai piedi.",
        ],
        neutral=[
            "Per le toilette giapponesi esistono spesso pantofole dedicate separate.",
            "Molti giapponesi indossano calzini puliti di ricambio quando vanno in visita.",
            "Nei ristoranti con tatami i posti a sedere sono su cuscini zabuton.",
        ],
    ),
    dict(
        id="train_etiquette",
        contexts=[
            "Sui treni e sulle metropolitane giapponesi è buona norma non parlare al telefono e tenere la suoneria in modalità silenziosa; nelle ore di punta alcune linee dispongono di carrozze riservate alle donne.",
            "In Giappone sui mezzi su rotaia si evita di telefonare e si tiene il telefono in modalità silenziosa; in alcune linee, nelle ore di punta, esistono vagoni riservati alle sole donne.",
        ],
        entail=[
            "Sui treni giapponesi non è educato parlare al telefono.",
            "Alcune linee giapponesi hanno carrozze riservate alle donne nelle ore di punta.",
            "In metropolitana in Giappone si tiene il telefono in silenzioso.",
        ],
        contradict=[
            "Sui treni giapponesi è normale fare lunghe telefonate ad alta voce.",
            "In Giappone non esistono carrozze riservate alle donne.",
            "Sui mezzi giapponesi è obbligatorio tenere la suoneria al massimo volume.",
        ],
        neutral=[
            "Nelle stazioni giapponesi le persone si mettono in fila ai punti segnati a terra.",
            "Molti pendolari giapponesi dormono durante il tragitto in treno.",
            "Mangiare è generalmente accettato sui treni a lunga percorrenza come lo Shinkansen.",
        ],
    ),
    # ------------------------------ CIBO E LOCALI ---------------------------
    dict(
        id="ichiran",
        contexts=[
            "La catena di ramen Ichiran è famosa per i banconi con postazioni singole separate da pannelli; si ordina tramite un distributore automatico di biglietti all'ingresso e molte sedi sono aperte 24 ore su 24.",
            "Da Ichiran, nota catena di ramen, si mangia in postazioni individuali divise da pannelli; l'ordine si effettua alla macchinetta dei biglietti all'entrata e diversi locali restano aperti 24 ore su 24.",
        ],
        entail=[
            "Da Ichiran si ordina tramite un distributore di biglietti.",
            "Ichiran ha postazioni singole separate da pannelli.",
            "Alcune sedi di Ichiran sono aperte 24 ore su 24.",
        ],
        contradict=[
            "Da Ichiran si ordina esclusivamente tramite camerieri al tavolo.",
            "Ichiran è noto per i grandi tavoli conviviali condivisi.",
            "Tutti i locali Ichiran chiudono alle 15:00.",
        ],
        neutral=[
            "Ichiran è specializzata nel ramen in stile tonkotsu di Fukuoka.",
            "Da Ichiran si può personalizzare la ricchezza del brodo e il piccante.",
            "Nei locali Ichiran il refill di noodles si chiama kaedama.",
        ],
    ),
    dict(
        id="tsukiji",
        contexts=[
            "Il mercato esterno di Tsukiji a Tokyo è attivo soprattutto al mattino: molte bancarelle aprono intorno alle 5:00 e gran parte dei negozi chiude già nel primo pomeriggio, verso le 14:00.",
            "A Tokyo il mercato esterno di Tsukiji vive di mattina: le bancarelle iniziano ad aprire verso le 5:00 e la maggior parte delle attività abbassa le serrande nel primo pomeriggio, attorno alle 14:00.",
        ],
        entail=[
            "Il mercato esterno di Tsukiji è più attivo al mattino.",
            "Molte bancarelle di Tsukiji aprono intorno alle 5:00.",
            "A Tsukiji gran parte dei negozi chiude verso le 14:00.",
        ],
        contradict=[
            "Il mercato esterno di Tsukiji apre solo la sera.",
            "A Tsukiji i negozi restano aperti fino a mezzanotte.",
            "Le bancarelle di Tsukiji aprono alle 11:00.",
        ],
        neutral=[
            "A Tsukiji si possono assaggiare spiedini di uova arrotolate tamagoyaki.",
            "L'asta dei tonni si svolge oggi al mercato di Toyosu.",
            "Il mercato di Tsukiji è raggiungibile con la linea Hibiya della metro.",
        ],
    ),
    dict(
        id="konbini",
        contexts=[
            "I konbini giapponesi come 7-Eleven, Lawson e FamilyMart sono aperti 24 ore su 24 e offrono cibo pronto, bevande, prelievo bancomat e servizi come la stampa di documenti e il pagamento delle bollette.",
            "In Giappone i minimarket (konbini) quali 7-Eleven, Lawson e FamilyMart restano aperti 24 ore su 24: vendono piatti pronti e bevande e offrono ATM, stampa documenti e pagamento delle bollette.",
        ],
        entail=[
            "I konbini giapponesi sono aperti 24 ore su 24.",
            "Nei konbini si possono pagare le bollette.",
            "Nei konbini giapponesi sono disponibili sportelli bancomat.",
        ],
        contradict=[
            "I konbini giapponesi chiudono tutti alle 18:00.",
            "Nei konbini non è possibile prelevare contanti.",
            "I konbini vendono solo prodotti per la casa e nessun alimento.",
        ],
        neutral=[
            "Molti konbini scaldano i piatti pronti alla cassa su richiesta.",
            "Gli onigiri dei konbini costano in media tra 120 e 200 yen.",
            "Nei konbini si possono ritirare i pacchi ordinati online.",
        ],
    ),
    # --------------------------- STAGIONI E CLIMA ---------------------------
    dict(
        id="sakura",
        contexts=[
            "A Tokyo e Kyoto la fioritura dei ciliegi (sakura) avviene in genere tra la fine di marzo e l'inizio di aprile, con una piena fioritura che dura circa una settimana.",
            "La stagione dei sakura a Tokyo e Kyoto cade solitamente tra fine marzo e i primi di aprile; il picco della fioritura dura all'incirca una settimana.",
        ],
        entail=[
            "A Tokyo i ciliegi fioriscono di solito tra fine marzo e inizio aprile.",
            "La piena fioritura dei sakura dura circa una settimana.",
            "A Kyoto la fioritura dei ciliegi avviene in genere a cavallo tra marzo e aprile.",
        ],
        contradict=[
            "A Tokyo i ciliegi fioriscono in pieno agosto.",
            "La piena fioritura dei sakura dura tre mesi.",
            "A Kyoto i sakura fioriscono a dicembre.",
        ],
        neutral=[
            "Durante la fioritura molti giapponesi fanno picnic hanami nei parchi.",
            "Il parco di Ueno è uno dei luoghi più famosi per ammirare i ciliegi a Tokyo.",
            "Le previsioni della fioritura vengono pubblicate ogni anno dagli enti meteo.",
        ],
    ),
    dict(
        id="momiji",
        contexts=[
            "Il foliage autunnale (momiji) a Kyoto raggiunge tipicamente il picco tra la metà e la fine di novembre, quando gli aceri si tingono di rosso nei giardini dei templi.",
            "A Kyoto il momento migliore per il momiji, il foliage autunnale degli aceri, è di norma tra metà e fine novembre, con i giardini dei templi che si colorano di rosso.",
        ],
        entail=[
            "A Kyoto il picco del foliage autunnale è tra metà e fine novembre.",
            "In autunno gli aceri di Kyoto diventano rossi.",
            "Il momiji indica il foliage autunnale degli aceri.",
        ],
        contradict=[
            "Il picco del momiji a Kyoto è a giugno.",
            "In autunno gli aceri giapponesi restano completamente verdi.",
            "Il foliage a Kyoto raggiunge il massimo a febbraio.",
        ],
        neutral=[
            "Alcuni templi di Kyoto organizzano aperture serali con illuminazioni.",
            "Il tempio Tofuku-ji è celebre per la vista sul foliage dal ponte Tsutenkyo.",
            "In autunno a Kyoto le temperature serali possono scendere sotto i 10 gradi.",
        ],
    ),
    dict(
        id="tsuyu",
        contexts=[
            "La stagione delle piogge (tsuyu) in gran parte del Giappone va all'incirca da inizio giugno a metà luglio, con piogge frequenti e alta umidità; Hokkaido ne è in gran parte esclusa.",
            "In quasi tutto il Giappone la tsuyu, la stagione delle piogge, dura più o meno da inizio giugno a metà luglio, portando precipitazioni frequenti e umidità elevata; Hokkaido resta in buona parte fuori da questo fenomeno.",
        ],
        entail=[
            "La stagione delle piogge in Giappone va circa da inizio giugno a metà luglio.",
            "Hokkaido è in gran parte esclusa dalla stagione delle piogge.",
            "Durante la tsuyu l'umidità è elevata.",
        ],
        contradict=[
            "La stagione delle piogge giapponese dura da dicembre a febbraio.",
            "Hokkaido è la regione più colpita dalla tsuyu.",
            "Durante la tsuyu il clima è secco e privo di piogge.",
        ],
        neutral=[
            "Durante la tsuyu fioriscono le ortensie in molti templi.",
            "In estate il Giappone può essere interessato dai tifoni.",
            "Molti hotel giapponesi prestano ombrelli gratuitamente agli ospiti.",
        ],
    ),
    dict(
        id="fuji_climbing",
        contexts=[
            "La stagione ufficiale di scalata del monte Fuji va all'incirca da inizio luglio a inizio settembre; fuori da questo periodo i sentieri e i rifugi sono in gran parte chiusi. Dal 2024 il sentiero Yoshida prevede un contributo di 2000 yen a persona.",
            "Il monte Fuji si può scalare ufficialmente da inizio luglio a inizio settembre, quando sentieri e rifugi sono operativi. Sul sentiero Yoshida, dal 2024, è richiesto un contributo di 2000 yen a persona.",
        ],
        entail=[
            "La stagione di scalata del Fuji va circa da luglio a inizio settembre.",
            "Sul sentiero Yoshida si paga un contributo di 2000 yen.",
            "Fuori stagione i rifugi del monte Fuji sono in gran parte chiusi.",
        ],
        contradict=[
            "Il monte Fuji si scala ufficialmente solo in inverno.",
            "L'accesso al sentiero Yoshida è completamente gratuito.",
            "I rifugi del Fuji restano aperti tutto l'anno.",
        ],
        neutral=[
            "Molti escursionisti salgono di notte per vedere l'alba dalla vetta del Fuji.",
            "Il monte Fuji è alto 3776 metri.",
            "Dalla regione dei cinque laghi si gode una vista celebre sul Fuji.",
        ],
    ),
    # ------------------------------ PRATICO/LOGISTICA -----------------------
    dict(
        id="cash_japan",
        contexts=[
            "Sebbene i pagamenti elettronici siano sempre più diffusi, in Giappone molti piccoli ristoranti, templi e negozi accettano solo contanti; è consigliabile avere sempre con sé alcune migliaia di yen.",
            "In Giappone il contante resta importante: nonostante la diffusione dei pagamenti digitali, parecchi piccoli ristoranti, templi e botteghe accettano soltanto contanti, quindi conviene portare con sé qualche migliaio di yen.",
        ],
        entail=[
            "In Giappone alcuni piccoli ristoranti accettano solo contanti.",
            "È consigliabile girare in Giappone con del contante.",
            "Molti templi giapponesi non accettano pagamenti elettronici.",
        ],
        contradict=[
            "In Giappone ogni negozio e tempio accetta la carta di credito.",
            "In Giappone il contante è stato completamente abolito.",
            "Nessun ristorante giapponese accetta pagamenti in contanti.",
        ],
        neutral=[
            "Gli ATM dei konbini 7-Eleven accettano le carte estere.",
            "In Giappone le monete arrivano fino al taglio da 500 yen.",
            "Molti negozi giapponesi offrono lo shopping tax-free ai turisti.",
        ],
    ),
    dict(
        id="plug_voltage",
        contexts=[
            "In Giappone le prese elettriche sono di tipo A con due lamelle piatte e la tensione è di 100 volt; i viaggiatori europei hanno bisogno di un adattatore.",
            "Le prese giapponesi sono di tipo A, a due lamelle piatte, con corrente a 100 volt: chi arriva dall'Europa deve munirsi di un adattatore.",
        ],
        entail=[
            "In Giappone la tensione elettrica è di 100 volt.",
            "Le prese giapponesi sono di tipo A.",
            "Un viaggiatore europeo ha bisogno di un adattatore in Giappone.",
        ],
        contradict=[
            "In Giappone la tensione è di 220 volt come in Europa.",
            "Le prese italiane funzionano in Giappone senza adattatore.",
            "Le prese giapponesi sono di tipo L a tre poli.",
        ],
        neutral=[
            "Molti hotel giapponesi mettono a disposizione prese USB accanto al letto.",
            "In Giappone la frequenza di rete varia tra 50 e 60 Hz a seconda della regione.",
            "Nei negozi di elettronica di Akihabara si trovano adattatori economici.",
        ],
    ),
    dict(
        id="visa_italy",
        contexts=[
            "I cittadini italiani possono entrare in Giappone per turismo senza visto per soggiorni fino a 90 giorni, con passaporto valido per tutta la durata del viaggio.",
            "Per motivi turistici i cittadini italiani non necessitano di visto per il Giappone fino a 90 giorni di permanenza; è richiesto un passaporto valido per l'intero soggiorno.",
        ],
        entail=[
            "Un turista italiano può stare in Giappone fino a 90 giorni senza visto.",
            "Per l'Italia non serve il visto turistico per il Giappone entro i 90 giorni.",
            "Serve un passaporto valido per tutta la durata del viaggio in Giappone.",
        ],
        contradict=[
            "I cittadini italiani devono sempre richiedere un visto per il Giappone.",
            "Il soggiorno senza visto per gli italiani in Giappone è di massimo 7 giorni.",
            "Per entrare in Giappone dall'Italia basta la carta d'identità.",
        ],
        neutral=[
            "All'arrivo in Giappone vengono rilevate le impronte digitali dei visitatori.",
            "Il modulo di ingresso si può precompilare online con Visit Japan Web.",
            "In Giappone bisogna portare sempre con sé il passaporto.",
        ],
    ),
    dict(
        id="luggage_forward",
        contexts=[
            "Il servizio di spedizione bagagli takkyubin permette di inviare le valigie da un hotel all'altro in Giappone, in genere con consegna il giorno successivo e un costo di circa 2000-3000 yen a collo.",
            "Con il takkyubin, il servizio giapponese di trasporto bagagli, si possono spedire le valigie tra hotel: la consegna avviene di solito il giorno dopo e il prezzo è di circa 2000-3000 yen a valigia.",
        ],
        entail=[
            "Il takkyubin consegna di solito le valigie il giorno successivo.",
            "Spedire una valigia con il takkyubin costa circa 2000-3000 yen.",
            "Con il takkyubin si possono spedire i bagagli da un hotel all'altro.",
        ],
        contradict=[
            "Il takkyubin impiega due settimane per consegnare una valigia.",
            "La spedizione di una valigia con takkyubin costa circa 20000 yen.",
            "In Giappone non esiste alcun servizio di spedizione bagagli tra hotel.",
        ],
        neutral=[
            "Il takkyubin si può richiedere anche dai konbini.",
            "Yamato Transport è uno dei principali operatori di takkyubin.",
            "Molti viaggiatori usano il takkyubin per spedire i bagagli in aeroporto.",
        ],
    ),
    dict(
        id="pocket_wifi",
        contexts=[
            "Per avere internet in viaggio in Giappone si può noleggiare un pocket WiFi o acquistare una SIM/eSIM dati; il ritiro del pocket WiFi è possibile direttamente nei banchi degli aeroporti e la riconsegna avviene spesso tramite una busta prepagata.",
            "In Giappone i turisti possono connettersi noleggiando un pocket WiFi oppure comprando una SIM o eSIM dati; il dispositivo si ritira ai banchi in aeroporto e si restituisce di frequente con una busta prepagata.",
        ],
        entail=[
            "Il pocket WiFi si può ritirare in aeroporto in Giappone.",
            "In Giappone si può usare una eSIM dati per avere internet.",
            "La riconsegna del pocket WiFi avviene spesso con una busta prepagata.",
        ],
        contradict=[
            "In Giappone è impossibile noleggiare un pocket WiFi.",
            "Il pocket WiFi si può restituire solo di persona nello stesso aeroporto di ritiro.",
            "Le eSIM non funzionano sul territorio giapponese.",
        ],
        neutral=[
            "Molte stazioni e konbini giapponesi offrono WiFi gratuito.",
            "Le eSIM turistiche per il Giappone si attivano tramite QR code.",
            "Un pocket WiFi può connettere più dispositivi contemporaneamente.",
        ],
    ),
    dict(
        id="ryokan",
        contexts=[
            "Nei ryokan tradizionali si dorme su futon stesi sul tatami e la tariffa include spesso la cena kaiseki e la colazione; il check-in avviene in genere tra le 15:00 e le 17:00.",
            "Un ryokan tradizionale prevede il pernottamento su futon posati sul tatami; nel prezzo sono spesso comprese la cena kaiseki e la colazione, e il check-in si effettua solitamente tra le 15:00 e le 17:00.",
        ],
        entail=[
            "Nei ryokan si dorme su futon sistemati sul tatami.",
            "La tariffa dei ryokan include spesso cena e colazione.",
            "Il check-in nei ryokan avviene in genere tra le 15:00 e le 17:00.",
        ],
        contradict=[
            "Nei ryokan si dorme esclusivamente su letti occidentali a molle.",
            "Nei ryokan i pasti non sono mai inclusi nella tariffa.",
            "Il check-in nei ryokan si fa solo dopo mezzanotte.",
        ],
        neutral=[
            "Nei ryokan gli ospiti indossano lo yukata fornito dalla struttura.",
            "Molti ryokan si trovano nelle località termali.",
            "La cena kaiseki è composta da numerose piccole portate stagionali.",
        ],
    ),
    dict(
        id="nara_deer",
        contexts=[
            "Nel parco di Nara vivono in libertà oltre mille cervi sika considerati messaggeri divini; i visitatori possono nutrirli solo con gli appositi cracker shika senbei venduti nel parco a circa 200 yen.",
            "Il parco di Nara ospita più di mille cervi sika in libertà, ritenuti messaggeri degli dei; è consentito dar loro da mangiare soltanto gli shika senbei, i cracker dedicati venduti nel parco a circa 200 yen.",
        ],
        entail=[
            "Nel parco di Nara vivono oltre mille cervi in libertà.",
            "I cervi di Nara si possono nutrire solo con gli shika senbei.",
            "I cracker per i cervi di Nara costano circa 200 yen.",
        ],
        contradict=[
            "Nel parco di Nara vivono soltanto una decina di cervi.",
            "A Nara si può dare ai cervi qualsiasi cibo portato da casa.",
            "Nel parco di Nara è vietato avvicinarsi ai cervi.",
        ],
        neutral=[
            "Alcuni cervi di Nara si inchinano per chiedere i cracker.",
            "Nara è stata la capitale del Giappone nell'VIII secolo.",
            "Il Grande Buddha del Todai-ji si trova vicino al parco di Nara.",
        ],
    ),
    dict(
        id="todaiji",
        contexts=[
            "Il tempio Todai-ji di Nara custodisce il Grande Buddha, una statua in bronzo alta circa 15 metri. L'ingresso alla Sala del Grande Buddha costa 800 yen per gli adulti.",
            "A Nara, il Todai-ji ospita il Daibutsu, il Grande Buddha in bronzo di circa 15 metri di altezza; per accedere alla Sala del Grande Buddha gli adulti pagano 800 yen.",
        ],
        entail=[
            "Il Grande Buddha del Todai-ji è alto circa 15 metri.",
            "L'ingresso alla Sala del Grande Buddha costa 800 yen per gli adulti.",
            "Il Todai-ji si trova a Nara.",
        ],
        contradict=[
            "Il Grande Buddha del Todai-ji è alto 50 centimetri.",
            "L'ingresso al Todai-ji è gratuito per gli adulti.",
            "Il Todai-ji si trova a Sapporo.",
        ],
        neutral=[
            "Una colonna del Todai-ji ha un foro che i visitatori provano ad attraversare.",
            "La sala principale del Todai-ji è uno degli edifici in legno più grandi al mondo.",
            "Il Todai-ji fu fondato nell'VIII secolo.",
        ],
    ),
    dict(
        id="arashiyama",
        contexts=[
            "La foresta di bambù di Arashiyama, a ovest di Kyoto, è accessibile gratuitamente e sempre aperta; il momento migliore per visitarla evitando la folla è la mattina presto.",
            "Ad Arashiyama, nella zona ovest di Kyoto, il sentiero nella foresta di bambù è gratuito e aperto a qualsiasi ora; per trovare poca gente conviene andarci al mattino presto.",
        ],
        entail=[
            "La foresta di bambù di Arashiyama è gratuita.",
            "Il sentiero del bambù di Arashiyama è sempre accessibile.",
            "Al mattino presto la foresta di bambù è meno affollata.",
        ],
        contradict=[
            "L'ingresso alla foresta di bambù di Arashiyama costa 1200 yen.",
            "La foresta di bambù di Arashiyama apre solo dalle 10:00 alle 16:00.",
            "Arashiyama si trova a est di Osaka, non vicino a Kyoto.",
        ],
        neutral=[
            "Ad Arashiyama si trova anche il tempio Tenryu-ji, patrimonio UNESCO.",
            "Il ponte Togetsukyo è uno dei simboli di Arashiyama.",
            "Ad Arashiyama c'è un parco dove osservare i macachi giapponesi.",
        ],
    ),
    dict(
        id="miyajima",
        contexts=[
            "L'isola di Miyajima, vicino a Hiroshima, è celebre per il torii galleggiante del santuario di Itsukushima. Il traghetto da Miyajimaguchi impiega circa 10 minuti e dal 2023 i visitatori pagano una tassa di ingresso di 100 yen.",
            "Miyajima, l'isola presso Hiroshima nota per il torii 'galleggiante' di Itsukushima, si raggiunge in circa 10 minuti di traghetto da Miyajimaguchi; dal 2023 è prevista una tassa turistica di 100 yen.",
        ],
        entail=[
            "Il traghetto per Miyajima impiega circa 10 minuti.",
            "Dal 2023 per visitare Miyajima si paga una tassa di 100 yen.",
            "Il santuario di Itsukushima si trova sull'isola di Miyajima.",
        ],
        contradict=[
            "Il traghetto per Miyajima dura circa tre ore.",
            "Non esiste alcuna tassa di ingresso per Miyajima.",
            "Il torii galleggiante di Itsukushima si trova a Okinawa.",
        ],
        neutral=[
            "Con la bassa marea si può camminare fino al torii di Itsukushima.",
            "A Miyajima vivono cervi in libertà come a Nara.",
            "Il momiji manju è il dolce tipico di Miyajima.",
        ],
    ),
    dict(
        id="hiroshima_peace",
        contexts=[
            "Il Museo Memoriale della Pace di Hiroshima ha un biglietto d'ingresso di 200 yen per gli adulti ed è aperto dalle 8:30, con orario di chiusura variabile a seconda della stagione.",
            "A Hiroshima, il Museo Memoriale della Pace costa 200 yen per gli adulti; apre alle 8:30 e l'orario di chiusura cambia in base alla stagione.",
        ],
        entail=[
            "Il Museo della Pace di Hiroshima costa 200 yen per gli adulti.",
            "Il Museo Memoriale della Pace apre alle 8:30.",
            "L'orario di chiusura del museo varia con la stagione.",
        ],
        contradict=[
            "Il Museo della Pace di Hiroshima costa 2000 yen per gli adulti.",
            "Il Museo Memoriale della Pace apre a mezzogiorno.",
            "Il museo di Hiroshima ha lo stesso orario di chiusura tutto l'anno.",
        ],
        neutral=[
            "Nel Parco della Pace si trova la Cupola della Bomba Atomica.",
            "Ogni 6 agosto a Hiroshima si tiene una cerimonia commemorativa.",
            "Hiroshima è famosa anche per l'okonomiyaki in stile locale.",
        ],
    ),
    dict(
        id="osaka_dotonbori",
        contexts=[
            "Dotonbori è il quartiere dei divertimenti di Osaka, famoso per le insegne luminose come quella del Glico Running Man e per lo street food; è più animato la sera e non prevede alcun biglietto d'ingresso.",
            "A Osaka, il quartiere di Dotonbori è celebre per le insegne al neon, tra cui il Glico Running Man, e per il cibo di strada; l'accesso è libero e la zona dà il meglio di sé la sera.",
        ],
        entail=[
            "A Dotonbori si trova l'insegna del Glico Running Man.",
            "L'accesso a Dotonbori è libero e gratuito.",
            "Dotonbori è particolarmente animato la sera.",
        ],
        contradict=[
            "Per entrare a Dotonbori si paga un biglietto di 500 yen.",
            "Dotonbori è una zona tranquilla che chiude alle 17:00.",
            "L'insegna del Glico Running Man si trova a Kyoto.",
        ],
        neutral=[
            "A Dotonbori si possono assaggiare i takoyaki, polpette di polpo.",
            "Il canale di Dotonbori è attraversato dal ponte Ebisubashi.",
            "Nelle vicinanze di Dotonbori si trova la via commerciale Shinsaibashi.",
        ],
    ),
    dict(
        id="universal_osaka",
        contexts=[
            "Gli Universal Studios Japan di Osaka richiedono il biglietto d'ingresso (Studio Pass); per l'area di Super Nintendo World nei giorni affollati serve anche un biglietto a orario (timed entry ticket), gratuito, da prenotare tramite l'app ufficiale.",
            "Per entrare agli Universal Studios Japan di Osaka serve lo Studio Pass; nei giorni di maggiore affluenza l'accesso a Super Nintendo World richiede in aggiunta un biglietto a orario gratuito, ottenibile dall'app ufficiale.",
        ],
        entail=[
            "Per gli Universal Studios Japan serve lo Studio Pass.",
            "Nei giorni affollati Super Nintendo World richiede un biglietto a orario.",
            "Il biglietto a orario per Super Nintendo World si prenota dall'app.",
        ],
        contradict=[
            "L'ingresso agli Universal Studios Japan è libero e gratuito.",
            "Super Nintendo World non richiede mai biglietti a orario.",
            "Il biglietto a orario per Super Nintendo World costa 5000 yen.",
        ],
        neutral=[
            "Agli Universal Studios Japan esistono pass Express a pagamento per saltare le code.",
            "Super Nintendo World include l'attrazione di Mario Kart.",
            "Il parco si raggiunge con la stazione Universal City della linea JR Yumesaki.",
        ],
    ),
    dict(
        id="kanazawa_kenrokuen",
        contexts=[
            "Il giardino Kenroku-en di Kanazawa, considerato uno dei tre giardini più belli del Giappone, costa 320 yen per gli adulti ed è aperto dalle 7:00 alle 18:00 da marzo a metà ottobre.",
            "A Kanazawa, il Kenroku-en — annoverato tra i tre giardini più celebri del Giappone — ha un ingresso di 320 yen per gli adulti; da marzo a metà ottobre gli orari vanno dalle 7:00 alle 18:00.",
        ],
        entail=[
            "Il Kenroku-en è considerato uno dei tre giardini più belli del Giappone.",
            "L'ingresso al Kenroku-en costa 320 yen per gli adulti.",
            "In estate il Kenroku-en apre alle 7:00.",
        ],
        contradict=[
            "L'ingresso al Kenroku-en costa 3200 yen.",
            "Il Kenroku-en è aperto solo di notte.",
            "Il Kenroku-en si trova a Fukuoka.",
        ],
        neutral=[
            "In inverno gli alberi del Kenroku-en vengono protetti con le corde yukitsuri.",
            "Accanto al Kenroku-en si trova il castello di Kanazawa.",
            "Kanazawa è famosa anche per il quartiere delle geishe Higashi Chaya.",
        ],
    ),
    dict(
        id="snow_monkey",
        contexts=[
            "Al Jigokudani Monkey Park, nella prefettura di Nagano, si possono osservare i macachi giapponesi immergersi nelle piscine termali; il parco è raggiungibile con una camminata di circa 30 minuti nel bosco e l'ingresso costa 800 yen per gli adulti.",
            "Nel Jigokudani Monkey Park (prefettura di Nagano) i macachi giapponesi fanno il bagno nelle vasche termali; per arrivarci si cammina circa mezz'ora nel bosco e il biglietto per gli adulti è di 800 yen.",
        ],
        entail=[
            "Al Jigokudani Monkey Park i macachi si immergono nelle acque termali.",
            "Per raggiungere il parco delle scimmie serve una camminata di circa 30 minuti.",
            "L'ingresso al Jigokudani Monkey Park costa 800 yen per gli adulti.",
        ],
        contradict=[
            "Il Jigokudani Monkey Park è raggiungibile in auto fino all'ingresso senza camminare.",
            "L'ingresso al parco delle scimmie di Nagano è gratuito.",
            "Al Jigokudani Monkey Park si osservano i panda giganti.",
        ],
        neutral=[
            "Le scimmie si vedono più facilmente in inverno, quando fa freddo.",
            "Il parco si trova vicino alla località termale di Yudanaka.",
            "Nagano ha ospitato le Olimpiadi invernali del 1998.",
        ],
    ),
    dict(
        id="sapporo_snowfes",
        contexts=[
            "Il Sapporo Snow Festival si tiene ogni anno a inizio febbraio nel parco Odori e in altre aree della città, con enormi sculture di neve e ghiaccio; l'ingresso alle aree del festival è gratuito.",
            "Ogni anno, ai primi di febbraio, Sapporo ospita lo Snow Festival: nel parco Odori e in altre zone vengono esposte gigantesche sculture di neve e ghiaccio e l'accesso alle aree è gratuito.",
        ],
        entail=[
            "Il Sapporo Snow Festival si svolge a inizio febbraio.",
            "L'ingresso alle aree del Sapporo Snow Festival è gratuito.",
            "Il festival della neve di Sapporo si tiene anche nel parco Odori.",
        ],
        contradict=[
            "Il Sapporo Snow Festival si tiene in piena estate, ad agosto.",
            "Per entrare alle aree del festival della neve si paga un biglietto di 4000 yen.",
            "Il festival della neve di Sapporo si svolge a Okinawa.",
        ],
        neutral=[
            "Durante il festival vengono organizzati anche scivoli di neve per bambini.",
            "A Sapporo è tipico il ramen al miso.",
            "Le sculture principali possono superare i dieci metri di altezza.",
        ],
    ),
    dict(
        id="gion",
        contexts=[
            "Nel quartiere di Gion a Kyoto, dal 2024 è vietato ai turisti entrare in alcuni vicoli privati; chi viola il divieto rischia una multa di 10000 yen. Le strade pubbliche principali come Hanamikoji restano accessibili.",
            "A Gion, il quartiere storico di Kyoto, dal 2024 alcuni vicoli privati sono off-limits per i turisti, con multe fino a 10000 yen per i trasgressori; le vie pubbliche principali, come Hanamikoji, restano percorribili.",
        ],
        entail=[
            "Dal 2024 alcuni vicoli privati di Gion sono vietati ai turisti.",
            "Chi entra nei vicoli vietati di Gion rischia una multa di 10000 yen.",
            "La via Hanamikoji resta accessibile ai visitatori.",
        ],
        contradict=[
            "A Gion i turisti possono entrare liberamente in tutti i vicoli privati.",
            "La multa per chi viola i divieti di Gion è di soli 10 yen.",
            "Dal 2024 l'intero quartiere di Gion è chiuso ai visitatori.",
        ],
        neutral=[
            "A Gion si trovano numerose case da tè storiche chiamate ochaya.",
            "Al tramonto è possibile incrociare maiko dirette agli appuntamenti.",
            "Il teatro Minamiza di Kyoto si trova ai margini di Gion.",
        ],
    ),
    dict(
        id="tokyo_skytree",
        contexts=[
            "La Tokyo Skytree è alta 634 metri ed è la torre più alta del Giappone; dispone di due osservatori, il Tembo Deck a 350 metri e la Tembo Galleria a 450 metri.",
            "Con i suoi 634 metri, la Tokyo Skytree è la torre più alta del Giappone e offre due piattaforme panoramiche: il Tembo Deck a quota 350 metri e la Tembo Galleria a 450 metri.",
        ],
        entail=[
            "La Tokyo Skytree è alta 634 metri.",
            "Il Tembo Deck si trova a 350 metri di altezza.",
            "La Tokyo Skytree ha due osservatori panoramici.",
        ],
        contradict=[
            "La Tokyo Skytree è alta 150 metri.",
            "La Tokyo Skytree ha un solo osservatorio.",
            "La Tembo Galleria si trova a 100 metri di altezza.",
        ],
        neutral=[
            "Alla base della Skytree si trova il centro commerciale Solamachi.",
            "La Skytree si illumina con colori diversi a seconda delle ricorrenze.",
            "La Tokyo Skytree ospita anche un acquario alla sua base.",
        ],
    ),
    dict(
        id="capsule_hotel",
        contexts=[
            "Nei capsule hotel giapponesi si dorme in capsule individuali disposte su due livelli; i bagni sono in comune, molte strutture hanno piani separati per uomini e donne e il prezzo medio per notte va da circa 3000 a 6000 yen.",
            "I capsule hotel in Giappone offrono capsule singole impilate su due file, con servizi igienici condivisi; spesso i piani sono divisi per genere e una notte costa in media tra 3000 e 6000 yen.",
        ],
        entail=[
            "Nei capsule hotel i bagni sono in comune.",
            "Una notte in capsule hotel costa in media tra 3000 e 6000 yen.",
            "Molti capsule hotel hanno piani separati per uomini e donne.",
        ],
        contradict=[
            "Ogni capsula di un capsule hotel dispone di bagno privato.",
            "Una notte in capsule hotel costa in media 50000 yen.",
            "Nei capsule hotel uomini e donne condividono sempre lo stesso piano.",
        ],
        neutral=[
            "Molti capsule hotel forniscono pigiama e articoli da toeletta.",
            "Alcuni capsule hotel moderni offrono capsule con TV integrata.",
            "I capsule hotel sono nati a Osaka alla fine degli anni Settanta.",
        ],
    ),
    dict(
        id="tax_free",
        contexts=[
            "In Giappone i turisti stranieri possono acquistare tax-free nei negozi aderenti per spese superiori a 5000 yen nello stesso giorno e nello stesso negozio, presentando il passaporto alla cassa.",
            "Lo shopping tax-free in Giappone è riservato ai turisti stranieri: serve una spesa minima di 5000 yen nello stesso negozio e nello stesso giorno, mostrando il passaporto al momento del pagamento.",
        ],
        entail=[
            "Per lo shopping tax-free in Giappone serve una spesa minima di 5000 yen.",
            "Il tax-free richiede di mostrare il passaporto alla cassa.",
            "Il tax-free vale per acquisti nello stesso negozio nello stesso giorno.",
        ],
        contradict=[
            "In Giappone il tax-free scatta da 100 yen di spesa.",
            "Per il tax-free non è necessario alcun documento.",
            "Il tax-free giapponese è riservato ai soli residenti in Giappone.",
        ],
        neutral=[
            "Alcuni grandi magazzini hanno banchi dedicati alle pratiche tax-free.",
            "L'IVA giapponese (consumption tax) è attualmente del 10%.",
            "I prodotti consumabili tax-free vengono sigillati in buste apposite.",
        ],
    ),
    dict(
        id="golden_week",
        contexts=[
            "La Golden Week giapponese è una serie di festività nazionali che cade tra la fine di aprile e l'inizio di maggio: in quei giorni treni e hotel sono estremamente affollati e i prezzi salgono sensibilmente.",
            "Tra fine aprile e inizio maggio si concentra la Golden Week, il periodo di festività nazionali giapponesi: trasporti e alloggi risultano molto affollati e le tariffe aumentano in modo marcato.",
        ],
        entail=[
            "La Golden Week cade tra fine aprile e inizio maggio.",
            "Durante la Golden Week gli hotel sono molto affollati.",
            "Nella Golden Week i prezzi di treni e alloggi aumentano.",
        ],
        contradict=[
            "La Golden Week giapponese si svolge a ottobre.",
            "Durante la Golden Week treni e hotel sono semivuoti.",
            "Nella Golden Week i prezzi degli alloggi crollano ai minimi annuali.",
        ],
        neutral=[
            "Molti giapponesi approfittano della Golden Week per viaggiare all'estero.",
            "Il 29 aprile in Giappone si celebra il giorno di Showa.",
            "Un'altra settimana molto affollata è l'Obon, a metà agosto.",
        ],
    ),
    dict(
        id="akihabara",
        contexts=[
            "Akihabara, il quartiere dell'elettronica e della cultura otaku di Tokyo, la domenica pomeriggio chiude al traffico la via principale Chuo-dori, che diventa un'isola pedonale.",
            "A Tokyo, il quartiere di Akihabara — regno dell'elettronica e della cultura otaku — trasforma la domenica pomeriggio la sua via principale, la Chuo-dori, in una zona pedonale chiusa alle auto.",
        ],
        entail=[
            "La domenica pomeriggio la Chuo-dori di Akihabara diventa pedonale.",
            "Akihabara è il quartiere dell'elettronica di Tokyo.",
            "La via principale di Akihabara si chiama Chuo-dori.",
        ],
        contradict=[
            "La Chuo-dori di Akihabara non viene mai chiusa al traffico.",
            "Akihabara è il quartiere finanziario di Tokyo, privo di negozi di elettronica.",
            "La via principale di Akihabara diventa pedonale solo il mercoledì mattina.",
        ],
        neutral=[
            "Ad Akihabara si trovano numerose sale giochi su più piani.",
            "Molti negozi di Akihabara sono specializzati in componenti elettronici usati.",
            "Ad Akihabara sono diffusi i maid café.",
        ],
    ),
    dict(
        id="onigiri_price",
        contexts=[
            "Mangiare in Giappone può essere economico: un pasto in una catena di gyudon come Yoshinoya o Sukiya costa circa 500-700 yen, mentre un ramen in un locale medio si aggira sui 1000 yen.",
            "In Giappone si può mangiare spendendo poco: una ciotola di gyudon da Yoshinoya o Sukiya costa sui 500-700 yen e un ramen in un ristorante medio costa attorno ai 1000 yen.",
        ],
        entail=[
            "Un gyudon da Yoshinoya costa circa 500-700 yen.",
            "Un ramen in un locale medio costa intorno ai 1000 yen.",
            "In Giappone si può mangiare in modo economico nelle catene di gyudon.",
        ],
        contradict=[
            "Un gyudon da Yoshinoya costa circa 7000 yen.",
            "Un ramen medio in Giappone costa almeno 10000 yen.",
            "In Giappone è impossibile mangiare con meno di 5000 yen a pasto.",
        ],
        neutral=[
            "Il gyudon è una ciotola di riso con striscioline di manzo.",
            "Molte catene giapponesi servono i pasti in pochi minuti.",
            "In Giappone l'acqua al ristorante viene servita gratuitamente.",
        ],
    ),
]

# ----------------------------------------------------------------------------
# 2. GENERAZIONE
# ----------------------------------------------------------------------------

LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}


def build_rows(seed: int):
    rng = random.Random(seed)
    rows = []
    fact_ids = [f["id"] for f in FACTS]

    for fact in FACTS:
        fid = fact["id"]
        for ctx in fact["contexts"]:
            for claim in fact["entail"]:
                rows.append((fid, ctx, claim, LABELS["entailment"], "entailment"))
            for claim in fact["contradict"]:
                rows.append((fid, ctx, claim, LABELS["contradiction"], "contradiction"))
            # Neutri "hard": stessa entità, attributo non dichiarato nel contesto
            for claim in fact["neutral"]:
                rows.append((fid, ctx, claim, LABELS["neutral"], "neutral_hard"))

        # Neutri "easy": claim pescato da un fatto diverso (entità scorrelata).
        # Ne aggiungiamo 2 per fatto (1 per contesto) per bilanciare le classi.
        other_ids = [x for x in fact_ids if x != fid]
        for ctx in fact["contexts"]:
            chosen_id = rng.choice(other_ids)
            other = next(f for f in FACTS if f["id"] == chosen_id)
            claim = rng.choice(other["entail"] + other["neutral"])
            rows.append((fid, ctx, claim, LABELS["neutral"], "neutral_easy"))

    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="factcheck_dataset_japan.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.seed)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_id", "context", "claim", "label", "claim_type"])
        writer.writerows(rows)

    from collections import Counter
    dist = Counter(r[3] for r in rows)
    print(f"Dataset scritto in {args.out}")
    print(f"Totale esempi: {len(rows)}")
    print(f"Fatti distinti (gruppi per lo split): {len(FACTS)}")
    print(f"Distribuzione classi: 0/contradiction={dist[0]}, 1/entailment={dist[1]}, 2/neutral={dist[2]}")


if __name__ == "__main__":
    main()
