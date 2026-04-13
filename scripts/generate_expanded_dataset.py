"""
Generate synthetic training data for all 9 classification classes.

Each class has domain-specific Italian-language template components and
sentence templates. Output goes to src/data/<class_slug>/synth_*.txt.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings


def _slugify(name: str) -> str:
    """Convert a label name to a directory slug (lowercase, underscores)."""
    import re

    slug = name.lower()
    slug = slug.replace("&", "").replace(",", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return re.sub(r"_+", "_", slug)

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data")

# ---------------------------------------------------------------------------
# Class-specific template components
# ---------------------------------------------------------------------------

CLASS_TEMPLATES = {
    # -----------------------------------------------------------------------
    # 0: Autonomous Driving and UVs
    # -----------------------------------------------------------------------
    "Autonomous Driving and UVs": {
        "components": {
            "vehicles": [
                "veicoli a guida autonoma di livello 4",
                "droni multirotore per consegne urbane",
                "UAV ad ala fissa per sorveglianza territoriale",
                "shuttle autonomi per trasporto passeggeri",
                "AGV (Automated Guided Vehicles) per logistica",
                "veicoli commerciali a guida semi-autonoma",
                "droni sottomarini per ispezioni offshore",
                "robot mobili autonomi per ambienti outdoor",
                "taxi volanti a decollo verticale (eVTOL)",
                "veicoli agricoli autonomi per precision farming",
                "minibus elettrici a guida autonoma",
                "droni per mappatura 3D del territorio",
                "piattaforme robotiche autonome per cantieri",
                "navi autonome per trasporto merci costiero",
                "rover autonomi per esplorazione ambientale",
            ],
            "sensors": [
                "sensori LiDAR a stato solido di ultima generazione",
                "telecamere stereo ad alta risoluzione con HDR",
                "radar a onde millimetriche a 77 GHz",
                "sensori a ultrasuoni per prossimita",
                "moduli GNSS RTK per posizionamento centimetrico",
                "sensori inerziali (IMU) a 9 assi",
                "telecamere termiche per visione notturna",
                "sensori di profondita Time-of-Flight",
                "antenne V2X per comunicazione veicolo-infrastruttura",
                "telecamere fisheye a 360 gradi",
                "sensori radar imaging SAR per droni",
                "moduli di visione stereoscopica con FPGA integrata",
            ],
            "tasks": [
                "navigazione autonoma in ambienti urbani complessi",
                "obstacle avoidance in tempo reale",
                "pianificazione del percorso con ottimizzazione multi-obiettivo",
                "SLAM visuale per localizzazione e mappatura simultanea",
                "fusione sensoriale multi-modale",
                "riconoscimento e classificazione di pedoni e ciclisti",
                "parcheggio autonomo in spazi ristretti",
                "lane detection e lane keeping su strade extraurbane",
                "gestione di incroci non semaforizzati",
                "atterraggio autonomo di precisione su piattaforme mobili",
                "coordinamento di flotte di veicoli autonomi",
                "decision-making in scenari a traffico misto",
            ],
            "environments": [
                "contesti urbani ad alta densita di traffico",
                "ambienti autostradali con corsie multiple",
                "zone portuali e aeroportuali",
                "magazzini logistici di grandi dimensioni",
                "aree agricole con terreno irregolare",
                "cantieri edili con ostacoli dinamici",
                "aree industriali con traffico misto uomo-macchina",
                "percorsi montani con condizioni meteo avverse",
                "zone pedonali con flussi variabili",
                "corridoi aerei urbani regolamentati",
            ],
        },
        "templates": [
            "Progetto di {task} per {vehicle} equipaggiato con {sensor} in {environment}. Il sistema integra algoritmi di percezione e pianificazione per garantire operazioni sicure ed efficienti.",
            "Sviluppo di un sistema di {task} basato su {sensor} per {vehicle}. La piattaforma opera in {environment} con capacita di adattamento in tempo reale alle condizioni operative.",
            "Realizzazione di {vehicle} con capacita di {task} mediante {sensor}. Il progetto prevede test estensivi in {environment} per la validazione della sicurezza funzionale.",
            "Implementazione di algoritmi di {task} per {vehicle} dotati di {sensor}. Il deployment avviene in {environment} con monitoraggio continuo delle performance.",
            "Piattaforma autonoma basata su {vehicle} per {task} in {environment}. L'architettura sensoriale comprende {sensor} con ridondanza hardware per safety-critical operations.",
            "Sistema avanzato di {task} per {vehicle} che sfrutta {sensor} per operare in {environment}. Il finanziamento copre sviluppo, prototipazione e certificazione secondo normative vigenti.",
            "Progetto di ricerca e sviluppo per {task} applicato a {vehicle}. La percezione ambientale si basa su {sensor}, con validazione in {environment} reali.",
            "Creazione di una flotta di {vehicle} con funzionalita di {task} tramite {sensor}. Le operazioni si svolgono in {environment} con supervisione remota.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 1: Enterprise AI
    # -----------------------------------------------------------------------
    "Enterprise AI": {
        "components": {
            "functions": [
                "gestione della supply chain",
                "ottimizzazione dei processi produttivi",
                "analisi predittiva delle vendite",
                "automazione del customer service",
                "gestione delle risorse umane",
                "business intelligence e reporting avanzato",
                "document understanding e classificazione automatica",
                "process mining e ottimizzazione dei workflow",
                "demand forecasting e pianificazione della produzione",
                "quality control automatizzato",
                "gestione del rischio operativo",
                "ottimizzazione della logistica e distribuzione",
                "automazione contabile e fiscale",
                "gestione intelligente dell'inventario",
                "monitoraggio e analisi delle performance aziendali",
            ],
            "techniques": [
                "predictive analytics con modelli ensemble",
                "Robotic Process Automation (RPA) potenziata da AI",
                "Natural Language Processing per analisi documentale",
                "computer vision per controllo qualita",
                "sistemi di raccomandazione per cross-selling",
                "modelli di ottimizzazione con programmazione lineare e AI",
                "anomaly detection per prevenzione frodi interne",
                "knowledge graph per gestione della conoscenza aziendale",
                "digital twin per simulazione di processi",
                "speech analytics per analisi conversazioni clienti",
                "modelli di churn prediction",
                "algoritmi di scheduling e resource allocation",
            ],
            "sectors": [
                "manifatturiero",
                "retail e grande distribuzione",
                "servizi finanziari",
                "pubblica amministrazione",
                "telecomunicazioni",
                "energia e utilities",
                "trasporti e logistica",
                "farmaceutico",
                "assicurativo",
                "food & beverage",
                "moda e lusso",
                "edilizia e costruzioni",
            ],
            "outcomes": [
                "riduzione dei costi operativi del 25-40%",
                "aumento dell'efficienza produttiva",
                "miglioramento della customer satisfaction",
                "accelerazione del time-to-market",
                "riduzione degli sprechi e delle inefficienze",
                "miglioramento della competitivita aziendale",
                "automazione di attivita ripetitive a basso valore",
                "supporto decisionale data-driven per il management",
                "ottimizzazione dell'allocazione delle risorse",
                "riduzione dei tempi di risposta al cliente",
            ],
        },
        "templates": [
            "Progetto di {function} nel settore {sector} basato su {technique}. L'obiettivo principale e {outcome}. Il sistema si integra con l'infrastruttura IT esistente.",
            "Implementazione di un sistema di {function} per aziende del settore {sector} utilizzando {technique}. Il progetto mira a {outcome} attraverso l'analisi di dati aziendali.",
            "Soluzione di intelligenza artificiale per {function} mediante {technique} applicata al settore {sector}. I risultati attesi includono {outcome}.",
            "Piattaforma AI per {function} dedicata al mercato {sector}. Il sistema sfrutta {technique} per garantire {outcome}.",
            "Sviluppo di una soluzione enterprise di {function} basata su {technique}. Il progetto si rivolge al settore {sector} con l'obiettivo di {outcome}.",
            "Sistema intelligente per {function} che utilizza {technique} per il settore {sector}. L'implementazione prevede {outcome} misurabile tramite KPI dedicati.",
            "Progetto di trasformazione digitale per {function} nel settore {sector}. La soluzione adotta {technique} per raggiungere {outcome}.",
            "Iniziativa di AI enterprise per {function} mediante {technique}. Destinata al settore {sector}, punta a {outcome} entro 12 mesi dal deployment.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 2: Environmental AI
    # -----------------------------------------------------------------------
    "Environmental AI": {
        "components": {
            "domains": [
                "monitoraggio della qualita dell'aria",
                "gestione sostenibile delle risorse idriche",
                "prevenzione del dissesto idrogeologico",
                "monitoraggio della biodiversita",
                "gestione intelligente dei rifiuti",
                "previsione e mitigazione degli incendi boschivi",
                "analisi dell'inquinamento acustico urbano",
                "monitoraggio delle emissioni di CO2",
                "tutela degli ecosistemi marini",
                "agricoltura di precisione sostenibile",
                "monitoraggio della deforestazione",
                "gestione delle aree protette",
                "analisi del cambiamento climatico",
                "bonifica di siti contaminati",
                "ottimizzazione energetica degli edifici",
            ],
            "sensing": [
                "reti di sensori IoT distribuiti sul territorio",
                "immagini satellitari multispettrali Sentinel-2",
                "stazioni meteo automatiche ad alta risoluzione",
                "sensori di qualita dell'aria a basso costo",
                "droni con camera iperspettrale",
                "boe oceanografiche con trasmissione dati in tempo reale",
                "reti di sensori acustici per monitoraggio fauna",
                "centraline di monitoraggio delle acque reflue",
                "radar meteorologici Doppler",
                "stazioni sismiche per monitoraggio geotecnico",
                "sensori soil moisture per monitoraggio del suolo",
                "camera trap con riconoscimento automatico delle specie",
            ],
            "ai_techniques": [
                "modelli di anomaly detection per allerta precoce",
                "reti neurali convoluzionali per classificazione immagini satellitari",
                "modelli di forecasting per previsione eventi estremi",
                "algoritmi di clustering per pattern ambientali",
                "deep learning per segmentazione del territorio",
                "modelli predittivi basati su serie temporali",
                "transfer learning per adattamento a nuovi ecosistemi",
                "reinforcement learning per ottimizzazione risorse",
                "NLP per analisi di report ambientali",
                "graph neural network per modellazione ecosistemi",
            ],
            "outcomes": [
                "allerta precoce per eventi naturali estremi",
                "riduzione dell'impatto ambientale delle attivita antropiche",
                "ottimizzazione dell'uso delle risorse naturali",
                "supporto alle politiche di sostenibilita ambientale",
                "miglioramento della resilienza territoriale",
                "protezione della biodiversita locale",
                "riduzione delle emissioni inquinanti",
                "prevenzione del degrado ambientale",
                "monitoraggio continuo dello stato di salute degli ecosistemi",
                "supporto alla transizione ecologica",
            ],
        },
        "templates": [
            "Progetto di {domain} basato su {ai_technique} che elabora dati provenienti da {sensing}. L'obiettivo e {outcome}.",
            "Sistema di intelligenza artificiale per {domain} mediante {ai_technique}. I dati vengono acquisiti tramite {sensing} per garantire {outcome}.",
            "Piattaforma di {domain} che integra {sensing} con {ai_technique}. Il progetto mira a {outcome} sul territorio regionale.",
            "Sviluppo di un sistema di {domain} basato su {ai_technique} alimentato da {sensing}. L'obiettivo finale e {outcome}.",
            "Soluzione AI per {domain} che utilizza {ai_technique} per analizzare dati da {sensing}. Il sistema contribuisce a {outcome}.",
            "Iniziativa di monitoraggio ambientale per {domain} tramite {ai_technique}. L'infrastruttura di raccolta dati include {sensing} per {outcome}.",
            "Progetto di ricerca applicata per {domain} con {ai_technique} e {sensing}. I risultati supportano {outcome} e la pianificazione territoriale.",
            "Sistema integrato di {domain} che combina {sensing} e {ai_technique} per {outcome}. Il finanziamento copre hardware, software e validazione sul campo.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 3: Fintech and Marketing
    # -----------------------------------------------------------------------
    "Fintech and Marketing": {
        "components": {
            "fin_tasks": [
                "credit scoring avanzato",
                "rilevamento frodi in tempo reale",
                "trading algoritmico ad alta frequenza",
                "risk management e stress testing",
                "antiriciclaggio (AML) automatizzato",
                "valutazione automatica del merito creditizio",
                "ottimizzazione del portafoglio investimenti",
                "analisi predittiva dei mercati finanziari",
                "pricing dinamico di prodotti assicurativi",
                "compliance normativa automatizzata (RegTech)",
                "gestione automatizzata dei sinistri",
                "analisi del rischio di credito per PMI",
            ],
            "mkt_tasks": [
                "segmentazione avanzata della clientela",
                "personalizzazione delle offerte commerciali",
                "dynamic pricing basato sulla domanda",
                "sentiment analysis sui social media",
                "lead scoring e qualificazione automatica",
                "attribution modeling multicanale",
                "ottimizzazione delle campagne pubblicitarie",
                "analisi predittiva del comportamento d'acquisto",
                "content recommendation personalizzata",
                "customer lifetime value prediction",
                "churn prediction e retention strategy",
                "market basket analysis per cross-selling",
            ],
            "data_types": [
                "transazioni finanziarie in tempo reale",
                "dati comportamentali degli utenti web e mobile",
                "serie temporali dei mercati finanziari",
                "dati CRM e storico interazioni cliente",
                "dati provenienti da open banking (PSD2)",
                "social media data e web scraping",
                "dati di navigazione e clickstream",
                "dati alternativi (satellite, geolocalizzazione)",
                "dati di mercato e indicatori macroeconomici",
                "feedback e recensioni dei clienti",
            ],
            "technologies": [
                "modelli di deep learning per sequenze temporali",
                "gradient boosting (XGBoost, LightGBM) per scoring",
                "reti neurali graph-based per reti di transazioni",
                "NLP per analisi di documenti finanziari",
                "reinforcement learning per strategie di trading",
                "autoencoders per anomaly detection",
                "transformer per previsione serie temporali",
                "federated learning per privacy dei dati bancari",
                "sistemi di raccomandazione collaborativi e content-based",
                "modelli bayesiani per quantificazione dell'incertezza",
            ],
        },
        "templates": [
            "Progetto di {fin_task} basato su {technology} che analizza {data_type}. Il sistema migliora la precisione decisionale e riduce i rischi operativi.",
            "Piattaforma di {mkt_task} mediante {technology} alimentata da {data_type}. La soluzione consente di ottimizzare le strategie commerciali in tempo reale.",
            "Sistema di {fin_task} e {mkt_task} integrati tramite {technology}. I dati elaborati includono {data_type} per una visione cliente a 360 gradi.",
            "Soluzione fintech per {fin_task} che utilizza {technology} su {data_type}. Il progetto rispetta le normative GDPR e PSD2 vigenti.",
            "Implementazione di {mkt_task} basata su {technology} con analisi di {data_type}. L'obiettivo e massimizzare il ROI delle attivita commerciali.",
            "Progetto di innovazione per {fin_task} mediante {technology}. Il sistema elabora {data_type} con latenza inferiore al secondo.",
            "Piattaforma AI per {mkt_task} nel settore finanziario che sfrutta {technology} e {data_type}. Il deployment avviene in ambiente cloud sicuro e certificato.",
            "Sistema intelligente per {fin_task} che combina {technology} e analisi di {data_type}. Il finanziamento copre sviluppo, certificazione e integrazione con i sistemi legacy.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 4: Generic use
    # -----------------------------------------------------------------------
    "Generic use": {
        "components": {
            "applications": [
                "classificazione automatica di documenti",
                "riconoscimento di immagini e oggetti",
                "analisi e sintesi di testi",
                "automazione di processi aziendali generici",
                "estrazione di informazioni da dati non strutturati",
                "sistemi di raccomandazione generici",
                "traduzione automatica multilingue",
                "analisi dei dati e reportistica automatizzata",
                "generazione automatica di contenuti",
                "ottimizzazione di processi decisionali",
                "digitalizzazione e archiviazione intelligente",
                "OCR e riconoscimento caratteri da documenti scansionati",
                "scheduling e pianificazione automatizzata",
                "analisi e visualizzazione di dati complessi",
            ],
            "contexts": [
                "trasformazione digitale delle organizzazioni",
                "innovazione tecnologica cross-settore",
                "modernizzazione dei processi operativi",
                "efficientamento delle attivita quotidiane",
                "supporto alla produttivita individuale e di team",
                "miglioramento dell'accessibilita dei servizi",
                "semplificazione di workflow complessi",
                "integrazione di sistemi informativi eterogenei",
                "potenziamento delle capacita analitiche",
                "democratizzazione dell'accesso all'intelligenza artificiale",
            ],
            "technologies": [
                "modelli di machine learning supervisionato",
                "algoritmi di deep learning pre-addestrati",
                "pipeline di data processing automatizzate",
                "API di intelligenza artificiale cloud-based",
                "framework open-source per ML (PyTorch, TensorFlow)",
                "modelli linguistici di grandi dimensioni (LLM)",
                "tecniche di transfer learning",
                "strumenti di AutoML per selezione automatica dei modelli",
                "piattaforme low-code per integrazione AI",
                "soluzioni di AI-as-a-Service",
            ],
            "outcomes": [
                "riduzione dei tempi di esecuzione delle attivita",
                "miglioramento della qualita dei risultati",
                "automazione di compiti ripetitivi",
                "supporto decisionale basato sui dati",
                "scalabilita e riproducibilita dei processi",
                "riduzione degli errori manuali",
                "accesso facilitato a strumenti avanzati di analisi",
                "aumento della produttivita complessiva",
                "creazione di valore attraverso l'analisi dei dati",
                "accelerazione dei processi di innovazione",
            ],
        },
        "templates": [
            "Progetto di {application} nell'ambito della {context}. Il sistema utilizza {technology} per {outcome}.",
            "Iniziativa di {application} finalizzata alla {context}. La soluzione adotta {technology} con l'obiettivo di {outcome}.",
            "Sviluppo di una piattaforma per {application} a supporto della {context}. Le tecnologie impiegate includono {technology} per garantire {outcome}.",
            "Implementazione di {application} tramite {technology} nel contesto della {context}. I benefici attesi comprendono {outcome}.",
            "Soluzione digitale per {application} che sfrutta {technology}. Il progetto si inserisce nella {context} e punta a {outcome}.",
            "Progetto trasversale di {application} basato su {technology}. L'intervento supporta la {context} con l'obiettivo di {outcome}.",
            "Sistema di {application} mediante {technology} per la {context}. Il finanziamento copre lo sviluppo, il testing e il deployment della soluzione per {outcome}.",
            "Piattaforma AI per {application} a supporto della {context}. L'approccio tecnologico si basa su {technology} per {outcome}.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 5: Healthcare AI
    # -----------------------------------------------------------------------
    "Healthcare AI": {
        "components": {
            "domains": [
                "diagnostica per immagini radiologiche",
                "oncologia computazionale",
                "cardiologia predittiva",
                "telemedicina e monitoraggio remoto",
                "neurologia e neuroimaging",
                "dermatologia assistita da AI",
                "oftalmologia e screening retinico",
                "chirurgia robotica assistita",
                "riabilitazione personalizzata",
                "gestione delle emergenze ospedaliere",
                "farmacogenomica e medicina personalizzata",
                "patologia digitale e istopatologia",
                "psichiatria computazionale",
                "odontoiatria digitale",
                "pediatria predittiva",
            ],
            "ai_tasks": [
                "diagnosi assistita da intelligenza artificiale",
                "analisi automatica di immagini mediche (TAC, MRI, radiografie)",
                "drug discovery e ottimizzazione molecolare",
                "monitoraggio continuo dei parametri vitali",
                "predizione del rischio clinico per pazienti critici",
                "segmentazione automatica di organi e lesioni",
                "analisi genomica e proteomica con ML",
                "supporto alla decisione clinica basato su evidence",
                "triage automatizzato in pronto soccorso",
                "pianificazione personalizzata della terapia",
                "analisi predittiva delle epidemie",
                "ottimizzazione dei flussi ospedalieri",
            ],
            "devices": [
                "dispositivi wearable per monitoraggio ECG continuo",
                "biosensori miniaturizzati per analisi point-of-care",
                "scanner TAC ad alta risoluzione con AI integrata",
                "ecografi portatili con assistenza AI",
                "sistemi robotici per chirurgia mininvasiva",
                "dispositivi di realta aumentata per assistenza chirurgica",
                "piattaforme di telemedicina con analisi AI",
                "sensori di monitoraggio ambientale ospedaliero",
                "dispositivi di somministrazione farmaci intelligenti",
                "sistemi di imaging digitale per patologia",
            ],
            "outcomes": [
                "diagnosi precoce con accuratezza superiore al 95%",
                "riduzione dei tempi di refertazione del 60%",
                "medicina personalizzata basata sul profilo genetico",
                "miglioramento degli esiti clinici dei pazienti",
                "riduzione dei ricoveri inappropriati",
                "ottimizzazione dell'allocazione delle risorse sanitarie",
                "supporto ai clinici per decisioni complesse",
                "monitoraggio proattivo dei pazienti cronici",
                "accelerazione della scoperta di nuovi farmaci",
                "riduzione degli errori diagnostici",
            ],
        },
        "templates": [
            "Progetto di {domain} basato su {ai_task} che utilizza {device}. L'obiettivo e {outcome}.",
            "Sistema di {ai_task} per {domain} alimentato da dati provenienti da {device}. Il progetto mira a {outcome}.",
            "Piattaforma AI per {domain} mediante {ai_task}. L'infrastruttura include {device} per garantire {outcome}.",
            "Sviluppo di {ai_task} applicato a {domain} con integrazione di {device}. I risultati attesi includono {outcome}.",
            "Soluzione di intelligenza artificiale per {domain} che combina {ai_task} e {device}. Il progetto persegue {outcome}.",
            "Implementazione clinica di {ai_task} per {domain}. Il sistema si interfaccia con {device} per {outcome}.",
            "Progetto di ricerca traslazionale per {domain} basato su {ai_task}. L'hardware di acquisizione comprende {device} per {outcome}.",
            "Iniziativa di sanita digitale per {domain} mediante {ai_task} e {device}. Il finanziamento copre sviluppo, validazione clinica e certificazione per {outcome}.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 6: Research
    # -----------------------------------------------------------------------
    "Research": {
        "components": {
            "areas": [
                "fisica delle particelle e cosmologia computazionale",
                "chimica computazionale e scienza dei materiali",
                "biologia computazionale e bioinformatica",
                "matematica applicata e ottimizzazione",
                "scienza dei dati e statistica avanzata",
                "ingegneria dei materiali avanzati",
                "astrofisica e analisi di dati astronomici",
                "geofisica e scienze della Terra",
                "scienze cognitive e neuroscienze computazionali",
                "climatologia e modellazione atmosferica",
                "genomica funzionale e trascrittomica",
                "meccanica quantistica computazionale",
                "fluidodinamica computazionale (CFD)",
                "linguistica computazionale",
                "scienze sociali computazionali",
            ],
            "methods": [
                "simulazioni Monte Carlo su larga scala",
                "reti neurali per risolvere equazioni differenziali (PINN)",
                "modelli generativi per scoperta di nuovi materiali",
                "reinforcement learning per ottimizzazione sperimentale",
                "graph neural network per strutture molecolari",
                "trasformatori per predizione di sequenze biologiche",
                "active learning per progettazione efficiente di esperimenti",
                "modelli di surrogato per simulazioni computazionalmente costose",
                "analisi topologica dei dati (TDA)",
                "variational inference per modelli probabilistici",
                "high-performance computing con acceleratori GPU",
                "Bayesian optimization per tuning di parametri sperimentali",
            ],
            "institutions": [
                "universita e centri di ricerca nazionali",
                "laboratori del CNR e dell'INFN",
                "centri di eccellenza europei (ERC)",
                "istituti di ricerca biomedica IRCCS",
                "consorzi interuniversitari di ricerca",
                "laboratori di ricerca industriale",
                "centri di supercalcolo (CINECA, GARR)",
                "fondazioni per la ricerca scientifica",
                "reti europee di infrastrutture di ricerca",
                "istituti di ricerca interdisciplinare",
            ],
            "outputs": [
                "pubblicazioni su riviste ad alto impatto",
                "brevetti e proprieta intellettuale",
                "dataset aperti per la comunita scientifica",
                "prototipi e proof-of-concept validati",
                "modelli computazionali riproducibili",
                "software open-source per la ricerca",
                "formazione di ricercatori specializzati",
                "contributi a conoscenze fondamentali del settore",
                "trasferimento tecnologico verso l'industria",
                "nuove metodologie sperimentali validate",
            ],
        },
        "templates": [
            "Progetto di ricerca in {area} che utilizza {method} presso {institution}. I risultati attesi includono {output}.",
            "Studio avanzato di {area} mediante {method} condotto da {institution}. L'obiettivo e produrre {output}.",
            "Ricerca fondamentale in {area} basata su {method}. Il progetto coinvolge {institution} e mira a {output}.",
            "Iniziativa di ricerca in {area} che impiega {method} con il supporto di {institution}. Le attivita producono {output}.",
            "Progetto scientifico in {area} con approccio basato su {method}. Realizzato da {institution}, il progetto genera {output}.",
            "Attivita di ricerca e innovazione in {area} mediante {method}. Il consorzio di {institution} mira a {output}.",
            "Programma di ricerca interdisciplinare in {area} che sfrutta {method}. Il progetto, coordinato da {institution}, produce {output}.",
            "Investigazione scientifica in {area} con {method} finanziata nell'ambito di programmi nazionali ed europei. Il lavoro presso {institution} genera {output}.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 7: Robotics and Industry
    # -----------------------------------------------------------------------
    "Robotics and Industry": {
        "components": {
            "processes": [
                "saldatura robotizzata ad alta precisione",
                "assemblaggio automatizzato di componenti",
                "controllo qualita visivo in linea di produzione",
                "movimentazione e pallettizzazione automatica",
                "confezionamento e packaging intelligente",
                "lavorazione CNC con ottimizzazione AI",
                "verniciatura robotizzata con visione artificiale",
                "ispezione non distruttiva automatizzata",
                "sorting e classificazione automatica di prodotti",
                "manutenzione predittiva di impianti industriali",
                "pick-and-place ad alta velocita",
                "sbavatura e finitura superficiale robotizzata",
                "dosaggio e miscelazione automatizzata",
                "taglio laser guidato da visione artificiale",
                "logistica intramurale automatizzata",
            ],
            "robots": [
                "bracci robotici antropomorfi a 6 assi",
                "robot collaborativi (cobot) per interazione uomo-macchina",
                "AGV e AMR per logistica di magazzino",
                "robot SCARA per assemblaggio ad alta velocita",
                "robot Delta per pick-and-place alimentare",
                "robot mobili autonomi con manipolatore integrato",
                "sistemi multi-robot coordinati",
                "esoscheletri industriali per supporto operatori",
                "robot di ispezione per ambienti pericolosi",
                "piattaforme robotiche modulari riconfigurabi",
                "droni industriali per ispezione dall'alto",
                "robot per saldatura orbitale automatica",
            ],
            "industry40": [
                "digital twin per simulazione e ottimizzazione",
                "predictive maintenance con sensori IoT",
                "smart factory con interconnessione M2M",
                "Industrial IoT (IIoT) per monitoraggio real-time",
                "edge computing per elaborazione dati on-premise",
                "sistemi MES/SCADA integrati con AI",
                "piattaforma Industry 4.0 per tracciabilita completa",
                "sistemi cyber-fisici per automazione avanzata",
                "cloud manufacturing per produzione distribuita",
                "gemello digitale dell'intera catena produttiva",
                "visione artificiale con deep learning on-edge",
                "realta aumentata per assistenza alla manutenzione",
            ],
            "sectors": [
                "automotive e componentistica",
                "food & beverage e industria alimentare",
                "meccanica di precisione",
                "elettronica e semiconduttori",
                "tessile e abbigliamento",
                "farmaceutico e cosmetico",
                "aerospaziale e difesa",
                "siderurgico e metallurgico",
                "legno e arredamento",
                "vetro e ceramica",
                "plastica e gomma",
                "cartario e packaging",
            ],
        },
        "templates": [
            "Progetto di {process} nel settore {sector} mediante {robot}. Il sistema integra {industry40} per ottimizzare produttivita e qualita.",
            "Implementazione di {robot} per {process} in ambito {sector}. L'architettura prevede {industry40} per il monitoraggio continuo.",
            "Soluzione di automazione industriale per {process} basata su {robot} con {industry40}. L'applicazione si rivolge al settore {sector}.",
            "Sviluppo di un sistema di {process} automatizzato tramite {robot} per il settore {sector}. L'integrazione con {industry40} garantisce efficienza e tracciabilita.",
            "Linea di produzione intelligente per {process} nel settore {sector} con {robot}. Il progetto adotta {industry40} per il controllo di processo.",
            "Cella robotizzata per {process} basata su {robot} con capacita di {industry40}. Destinata al settore {sector} per migliorare qualita e throughput.",
            "Progetto di innovazione industriale per {process} mediante {robot} e {industry40}. Il finanziamento copre hardware, integrazione e commissioning nel settore {sector}.",
            "Sistema robotico avanzato per {process} nel comparto {sector}. La soluzione combina {robot} e {industry40} per automazione end-to-end.",
        ],
        "count": 200,
    },
    # -----------------------------------------------------------------------
    # 8: Virtual assistants
    # -----------------------------------------------------------------------
    "Virtual assistants": {
        "components": {
            "types": [
                "chatbot conversazionale multicanale",
                "assistente vocale intelligente",
                "concierge digitale per strutture ricettive",
                "helpdesk AI per supporto tecnico",
                "assistente virtuale per la pubblica amministrazione",
                "agente conversazionale per e-commerce",
                "assistente digitale per servizi bancari",
                "bot di triage per servizi sanitari",
                "tutor virtuale per formazione personalizzata",
                "assistente AI per gestione appuntamenti",
                "receptionist virtuale per uffici",
                "assistente conversazionale per risorse umane",
                "companion digitale per anziani",
                "assistente vocale per domotica",
                "agente AI per prenotazioni e ticketing",
            ],
            "nlp_tasks": [
                "intent recognition con classificazione multi-label",
                "dialogue management con gestione del contesto",
                "sentiment analysis e emotion detection in tempo reale",
                "entity extraction e slot filling",
                "generazione di risposte naturali (NLG)",
                "comprensione del linguaggio naturale (NLU) in italiano",
                "risoluzione di ambiguita e coreference",
                "question answering su knowledge base aziendale",
                "summarization delle conversazioni precedenti",
                "language detection e switching multilingue",
                "speech-to-text e text-to-speech in italiano",
                "gestione di conversazioni multi-turno complesse",
            ],
            "contexts": [
                "customer service per aziende di telecomunicazioni",
                "assistenza clienti nel settore bancario",
                "supporto cittadini per servizi comunali",
                "guida turistica interattiva per musei e citta d'arte",
                "assistenza alla vendita per e-commerce",
                "helpdesk interno per dipendenti aziendali",
                "accoglienza pazienti in strutture sanitarie",
                "supporto studenti in piattaforme di e-learning",
                "assistenza tecnica per prodotti software",
                "informazioni e prenotazioni per eventi culturali",
                "supporto alla navigazione in app mobile",
                "consulenza automatizzata per servizi assicurativi",
            ],
            "technologies": [
                "modelli linguistici di grandi dimensioni (LLM) fine-tuned",
                "architetture Retrieval-Augmented Generation (RAG)",
                "framework di dialogue management (Rasa, Dialogflow)",
                "motori TTS/STT per interazione vocale naturale",
                "knowledge graph per risposte contestuali",
                "tecniche di prompt engineering avanzato",
                "sistemi di memoria conversazionale a lungo termine",
                "integrazione con API esterne e sistemi CRM",
                "pipeline di moderazione e safety dei contenuti",
                "analisi real-time del sentiment durante la conversazione",
            ],
        },
        "templates": [
            "Progetto di {type} per {context} basato su {technology}. Il sistema supporta {nlp_task} per garantire interazioni naturali e personalizzate.",
            "Sviluppo di un {type} mediante {technology} per {context}. Le funzionalita includono {nlp_task} con supporto multicanale.",
            "Implementazione di {type} con capacita di {nlp_task} tramite {technology}. La soluzione e destinata a {context}.",
            "Piattaforma di {type} per {context} che sfrutta {technology}. Il sistema integra {nlp_task} per conversazioni fluide e contestuali.",
            "Sistema conversazionale basato su {type} per {context}. L'architettura utilizza {technology} con funzionalita di {nlp_task}.",
            "{type} intelligente per {context} alimentato da {technology}. Le capacita di {nlp_task} consentono interazioni efficaci 24/7.",
            "Soluzione di {type} per {context} mediante {technology}. Il progetto implementa {nlp_task} con continuous improvement basato sul feedback utente.",
            "Progetto di {type} avanzato per {context} che integra {technology} e {nlp_task}. Il finanziamento copre sviluppo, training del modello e integrazione nei canali di contatto.",
        ],
        "count": 200,
    },
}


def generate_all():
    total = 0

    for class_name, config in CLASS_TEMPLATES.items():
        slug = _slugify(class_name)
        out_dir = os.path.join(BASE_DATA_PATH, slug)
        os.makedirs(out_dir, exist_ok=True)

        components = config["components"]
        templates = config["templates"]
        count = config["count"]

        for i in range(count):
            # Pick one random value from each component list
            chosen = {}
            for key, values in components.items():
                chosen[key] = random.choice(values)

            # Pick a random template and fill it
            template = random.choice(templates)
            try:
                text = template.format(**chosen)
            except KeyError:
                # Template references a key not in chosen — pick another template
                for t in templates:
                    try:
                        text = t.format(**chosen)
                        break
                    except KeyError:
                        continue
                else:
                    continue

            filename = f"synth_{slug}_{i:04d}.txt"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"  {class_name}: {count} synthetic files in {slug}/")
        total += count

    print(f"Total: {total} synthetic files generated")


if __name__ == "__main__":
    generate_all()
