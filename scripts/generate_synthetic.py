"""
Generate a small set of curated synthetic examples for each of the 9 classes.

Each example is a realistic Italian project description written by hand.
Output: src/data/<class_slug>/synthetic_<slug>_<i>.txt
"""

import os
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

CLASS_EXAMPLES = {
    "Autonomous Driving and UVs": [
        "Sviluppo di un sistema di guida autonoma di livello 4 per shuttle elettrici destinati al trasporto passeggeri in aree urbane pedonali, con sensori LiDAR e telecamere stereo.",
        "Progetto di drone multirotore per consegne last-mile in aree metropolitane, dotato di sistemi di obstacle avoidance e navigazione GPS RTK ad alta precisione.",
        "Implementazione di algoritmi di SLAM visuale per la navigazione autonoma di AGV in magazzini logistici di grandi dimensioni con traffico misto uomo-macchina.",
        "Realizzazione di un veicolo agricolo autonomo per precision farming, equipaggiato con sensori multispettrali e capacita di pianificazione del percorso su terreno irregolare.",
        "Piattaforma di fusione sensoriale multi-modale per veicoli a guida autonoma che integra LiDAR, radar a 77 GHz e telecamere per la percezione a 360 gradi dell'ambiente circostante.",
        "Sviluppo di un sistema di atterraggio autonomo di precisione per droni su piattaforme mobili, basato su visione artificiale e algoritmi di tracking in tempo reale.",
        "Progetto di flotta coordinata di UAV per sorveglianza territoriale con pianificazione cooperativa del percorso e comunicazione V2V per copertura ottimale dell'area.",
    ],
    "Enterprise AI": [
        "Implementazione di algoritmi di Machine Learning per la manutenzione predittiva dei macchinari industriali, riducendo i tempi di inattivita del 35% nel settore manifatturiero.",
        "Piattaforma cloud integrata con modelli di AI generativa per la creazione automatizzata di contenuti di marketing personalizzati per il settore retail.",
        "Integrazione di modelli predittivi ML per l'analisi dei dati aziendali e l'ottimizzazione della supply chain, con riduzione dei costi logistici del 20%.",
        "Sistema di business intelligence potenziato da AI per l'analisi predittiva delle vendite e la pianificazione della produzione nel settore food & beverage.",
        "Progetto di automazione dei processi aziendali tramite RPA potenziata da intelligenza artificiale cognitiva per la gestione documentale e contabile.",
        "Soluzione di document understanding basata su NLP per la classificazione automatica di contratti e fatture nel settore assicurativo.",
        "Implementazione di un sistema di process mining con AI per identificare e ottimizzare i colli di bottiglia nei workflow aziendali del settore telecomunicazioni.",
    ],
    "Environmental AI": [
        "Sistema di monitoraggio della qualita dell'aria basato su reti di sensori IoT e modelli di deep learning per la previsione dei livelli di inquinamento in aree urbane.",
        "Progetto di prevenzione incendi boschivi tramite analisi di immagini satellitari multispettrali e algoritmi di anomaly detection per allerta precoce.",
        "Piattaforma AI per la gestione sostenibile delle risorse idriche che integra dati da stazioni meteo e sensori di portata per ottimizzare la distribuzione dell'acqua.",
        "Sviluppo di un sistema di monitoraggio della biodiversita basato su camera trap con riconoscimento automatico delle specie tramite reti neurali convoluzionali.",
        "Progetto di previsione del dissesto idrogeologico mediante modelli predittivi che analizzano dati pluviometrici, geotecnici e immagini satellitari radar.",
        "Sistema intelligente per la gestione dei rifiuti urbani basato su AI che ottimizza i percorsi di raccolta e prevede i volumi di conferimento per area.",
        "Monitoraggio delle emissioni di gas serra tramite sensori distribuiti e modelli di machine learning per il supporto alle politiche di decarbonizzazione.",
    ],
    "Fintech and Marketing": [
        "Sistema di credit scoring avanzato basato su gradient boosting e dati alternativi per la valutazione del merito creditizio di PMI e startup innovative.",
        "Piattaforma di rilevamento frodi in tempo reale per transazioni bancarie che utilizza autoencoders e graph neural networks per identificare pattern sospetti.",
        "Progetto di segmentazione avanzata della clientela per una compagnia assicurativa mediante algoritmi di clustering e modelli predittivi del customer lifetime value.",
        "Implementazione di un sistema di dynamic pricing basato su reinforcement learning per l'ottimizzazione dei ricavi nel settore e-commerce.",
        "Soluzione di sentiment analysis sui social media per il monitoraggio in tempo reale della reputazione del brand tramite modelli NLP pre-addestrati su testi italiani.",
        "Sistema di trading algoritmico basato su transformer per la previsione di serie temporali finanziarie con gestione automatizzata del rischio.",
        "Piattaforma di marketing automation con AI per personalizzazione delle campagne email e ottimizzazione del conversion rate tramite A/B testing intelligente.",
    ],
    "Generic use": [
        "Sviluppo di un sistema di classificazione automatica di documenti basato su modelli di machine learning supervisionato per la digitalizzazione degli archivi aziendali.",
        "Piattaforma di analisi dei dati con AI per la generazione automatica di report e dashboard, accessibile tramite interfaccia low-code per utenti non tecnici.",
        "Implementazione di un sistema OCR intelligente per il riconoscimento e l'estrazione di informazioni da documenti scansionati di varia tipologia.",
        "Progetto di automazione dei processi tramite pipeline di data processing con modelli di machine learning per la riduzione degli errori manuali.",
        "Sistema di traduzione automatica multilingue basato su modelli linguistici di grandi dimensioni per supportare la comunicazione internazionale dell'organizzazione.",
        "Sviluppo di una soluzione di AI generativa per la creazione automatizzata di contenuti testuali a supporto della produttivita aziendale.",
        "Piattaforma di scheduling e pianificazione automatizzata basata su algoritmi di ottimizzazione per migliorare l'allocazione delle risorse organizzative.",
    ],
    "Healthcare AI": [
        "Sviluppo di un sistema esperto basato su reti neurali per il supporto alla diagnosi radiologica, con analisi automatica di immagini TAC e MRI.",
        "Progetto di telemedicina con monitoraggio remoto dei parametri vitali tramite dispositivi wearable e algoritmi di predizione del rischio clinico.",
        "Piattaforma di drug discovery basata su deep learning per l'identificazione di nuovi composti farmacologici attivi tramite analisi di strutture molecolari.",
        "Sistema di triage automatizzato per il pronto soccorso basato su NLP e machine learning per la classificazione della gravita dei pazienti.",
        "Implementazione di algoritmi di segmentazione automatica di organi e lesioni su immagini mediche per supporto alla pianificazione chirurgica.",
        "Progetto di medicina personalizzata basato su analisi genomica con modelli di machine learning per l'ottimizzazione dei protocolli terapeutici oncologici.",
        "Sistema di monitoraggio continuo dei pazienti cronici tramite biosensori e modelli predittivi per la prevenzione delle riacutizzazioni.",
    ],
    "Research": [
        "Progetto di ricerca in fisica delle particelle che utilizza reti neurali per l'analisi dei dati del rivelatore, condotto presso laboratori INFN.",
        "Studio di chimica computazionale che impiega modelli generativi per la scoperta di nuovi materiali con proprieta specifiche per applicazioni energetiche.",
        "Ricerca in bioinformatica basata su graph neural network per la predizione della struttura e della funzione delle proteine.",
        "Progetto scientifico di climatologia computazionale che sfrutta simulazioni Monte Carlo e modelli di surrogato per la previsione di eventi climatici estremi.",
        "Attivita di ricerca in neuroscienze computazionali mediante reti neurali per la modellazione dei processi cognitivi e l'analisi di segnali EEG.",
        "Programma di ricerca interdisciplinare in fluidodinamica computazionale che utilizza physics-informed neural networks per la risoluzione di equazioni di Navier-Stokes.",
        "Investigazione scientifica in astrofisica con modelli di deep learning per l'analisi automatica di survey astronomici e la classificazione di galassie.",
    ],
    "Robotics and Industry": [
        "Cella robotizzata per saldatura automatizzata ad alta precisione con bracci antropomorfi a 6 assi e sistema di visione artificiale per il settore automotive.",
        "Implementazione di robot collaborativi (cobot) per l'assemblaggio di componenti elettronici con integrazione in linea di produzione smart factory.",
        "Sistema di controllo qualita visivo in linea basato su deep learning e telecamere industriali per il rilevamento di difetti nel settore ceramico.",
        "Progetto di logistica intramurale con flotta di AMR (Autonomous Mobile Robots) coordinati tramite digital twin per l'ottimizzazione dei flussi di magazzino.",
        "Piattaforma di manutenzione predittiva Industry 4.0 basata su sensori IoT e modelli di anomaly detection per impianti di produzione alimentare.",
        "Linea di confezionamento automatizzato con robot Delta per pick-and-place ad alta velocita e sistema di visione per il settore farmaceutico.",
        "Implementazione di digital twin dell'intera catena produttiva con simulazione in tempo reale per l'ottimizzazione dei parametri di processo nel settore meccanico.",
    ],
    "Virtual assistants": [
        "Sviluppo di un chatbot conversazionale multicanale per il customer service di un operatore telefonico, con comprensione del linguaggio naturale in italiano.",
        "Implementazione di un assistente vocale intelligente per la domotica basato su riconoscimento vocale e dialogo multi-turno in lingua italiana.",
        "Realizzazione di un concierge digitale per strutture ricettive turistiche con capacita di prenotazione, informazioni e raccomandazioni personalizzate.",
        "Progetto di helpdesk AI per il supporto tecnico interno aziendale basato su architettura RAG con knowledge base documentale.",
        "Assistente virtuale per la pubblica amministrazione con funzionalita di guida ai servizi comunali, prenotazione appuntamenti e compilazione modulistica.",
        "Tutor virtuale per piattaforme di e-learning con capacita di adattamento al livello dello studente e generazione di esercizi personalizzati.",
        "Agente conversazionale per e-commerce con funzionalita di product recommendation, gestione ordini e assistenza post-vendita in tempo reale.",
    ],
}


def generate():
    total = 0
    for class_name, examples in CLASS_EXAMPLES.items():
        slug = _slugify(class_name)
        out_dir = os.path.join(BASE_DATA_PATH, slug)
        os.makedirs(out_dir, exist_ok=True)

        for i, text in enumerate(examples):
            filename = f"synthetic_{slug}_{i}.txt"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"  {class_name}: {len(examples)} examples in {slug}/")
        total += len(examples)

    print(f"Total: {total} synthetic examples generated")


if __name__ == "__main__":
    generate()
