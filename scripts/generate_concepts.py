"""
Generate concept/definition examples for each of the 9 classes.

Each example is a conceptual description of a key topic within the class domain,
written in Italian. These supplement the project-style descriptions with definitional content.
Output: src/data/<class_slug>/concept_<slug>_<i>.txt
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

CLASS_CONCEPTS = {
    "Autonomous Driving and UVs": [
        "La guida autonoma di livello 4 consente al veicolo di operare senza intervento umano in condizioni operative definite, utilizzando sensori LiDAR, radar e telecamere per percepire l'ambiente circostante.",
        "Il SLAM (Simultaneous Localization and Mapping) e una tecnica fondamentale per i veicoli autonomi che consente di costruire una mappa dell'ambiente e localizzare il veicolo al suo interno in tempo reale.",
        "La fusione sensoriale combina dati provenienti da LiDAR, radar, telecamere e GPS per creare una rappresentazione robusta e ridondante dell'ambiente, essenziale per la sicurezza dei veicoli autonomi.",
        "I droni o UAV (Unmanned Aerial Vehicles) sono aeromobili a pilotaggio remoto o autonomo utilizzati per sorveglianza, consegne, mappatura del territorio e ispezioni in ambienti pericolosi.",
        "La comunicazione V2X (Vehicle-to-Everything) permette ai veicoli autonomi di scambiare informazioni con altri veicoli, infrastrutture stradali e pedoni per migliorare la sicurezza e l'efficienza del traffico.",
        "Gli AGV (Automated Guided Vehicles) sono veicoli a guida automatica utilizzati nella logistica industriale per il trasporto di materiali lungo percorsi predefiniti o dinamici all'interno di stabilimenti.",
        "La pianificazione del percorso (path planning) per veicoli autonomi utilizza algoritmi come A*, RRT e algoritmi basati su reinforcement learning per calcolare traiettorie sicure ed efficienti.",
    ],
    "Enterprise AI": [
        "Il Machine Learning aziendale si riferisce all'applicazione di algoritmi di apprendimento automatico per ottimizzare processi di business come vendite, logistica, risorse umane e customer service.",
        "La Robotic Process Automation (RPA) potenziata da AI combina l'automazione di attivita ripetitive con capacita cognitive come il riconoscimento di documenti e la comprensione del linguaggio naturale.",
        "Il process mining utilizza algoritmi di data mining per analizzare i log dei sistemi informativi aziendali e scoprire, monitorare e migliorare i processi reali dell'organizzazione.",
        "I sistemi di raccomandazione enterprise utilizzano tecniche di collaborative filtering e content-based filtering per suggerire prodotti, servizi o azioni ai clienti o ai decisori aziendali.",
        "Il digital twin aziendale e una replica virtuale di un processo o sistema fisico che utilizza dati in tempo reale e modelli AI per simulare, prevedere e ottimizzare le operazioni.",
        "La demand forecasting con AI utilizza modelli di serie temporali e deep learning per prevedere la domanda futura di prodotti, consentendo una pianificazione produttiva e logistica ottimale.",
        "Il knowledge graph aziendale organizza la conoscenza dell'organizzazione in un grafo di entita e relazioni, interrogabile tramite AI per supporto decisionale e gestione della conoscenza.",
    ],
    "Environmental AI": [
        "Il monitoraggio ambientale con AI utilizza reti di sensori IoT e algoritmi di machine learning per rilevare in tempo reale variazioni nei parametri di qualita dell'aria, acqua e suolo.",
        "Il remote sensing applicato all'ambiente combina immagini satellitari multispettrali con deep learning per classificare l'uso del suolo, monitorare la deforestazione e tracciare il cambiamento climatico.",
        "La previsione di eventi meteorologici estremi con AI sfrutta modelli di forecasting basati su reti neurali per fornire allerte precoci su alluvioni, frane e ondate di calore.",
        "L'agricoltura di precisione sostenibile utilizza droni, sensori e modelli predittivi per ottimizzare l'uso di acqua, fertilizzanti e pesticidi, riducendo l'impatto ambientale delle coltivazioni.",
        "Il monitoraggio della biodiversita con AI impiega camera trap, sensori acustici e algoritmi di classificazione per censire le specie animali e vegetali presenti in un ecosistema.",
        "La gestione intelligente dei rifiuti usa sensori di riempimento sui cassonetti e algoritmi di ottimizzazione dei percorsi per ridurre i costi e le emissioni dei veicoli di raccolta.",
        "I modelli climatici potenziati da AI integrano simulazioni fisiche con reti neurali per migliorare la risoluzione e l'accuratezza delle proiezioni sul cambiamento climatico globale.",
    ],
    "Fintech and Marketing": [
        "Il credit scoring con AI utilizza modelli di gradient boosting e reti neurali per valutare il rischio creditizio di un richiedente, integrando dati tradizionali e alternativi.",
        "La fraud detection in ambito finanziario impiega algoritmi di anomaly detection e graph neural networks per identificare transazioni sospette in tempo reale sui circuiti di pagamento.",
        "Il trading algoritmico utilizza modelli di deep learning per analizzare dati di mercato ad alta frequenza e eseguire operazioni di compravendita automatizzate con strategie ottimizzate.",
        "La segmentazione della clientela con AI applica tecniche di clustering e modelli predittivi per suddividere i clienti in gruppi omogenei e personalizzare le strategie commerciali.",
        "Il dynamic pricing basato su AI adatta automaticamente i prezzi dei prodotti in funzione della domanda, della concorrenza e del comportamento degli utenti per massimizzare i ricavi.",
        "Il RegTech (Regulatory Technology) utilizza NLP e machine learning per automatizzare i controlli di conformita normativa, antiriciclaggio e segnalazione alle autorita di vigilanza.",
        "L'attribution modeling multicanale con AI attribuisce il merito delle conversioni ai diversi touchpoint del customer journey per ottimizzare l'allocazione del budget pubblicitario.",
    ],
    "Generic use": [
        "Il Machine Learning e una branca dell'intelligenza artificiale che consente ai sistemi di apprendere dai dati e migliorare le proprie prestazioni senza essere programmati esplicitamente per ogni compito.",
        "Il Deep Learning utilizza reti neurali artificiali con strati multipli per modellare astrazioni di alto livello nei dati, applicabile a immagini, testo, audio e dati tabulari.",
        "L'elaborazione del linguaggio naturale (NLP) permette ai computer di comprendere, interpretare e generare testo in linguaggio umano, abilitando applicazioni come traduzione, classificazione e chatbot.",
        "Il transfer learning consente di riutilizzare modelli pre-addestrati su grandi dataset per nuovi compiti con pochi dati, accelerando lo sviluppo di applicazioni AI in diversi domini.",
        "L'AutoML automatizza il processo di selezione, configurazione e ottimizzazione dei modelli di machine learning, rendendo l'AI accessibile anche a utenti senza competenze specialistiche.",
        "I modelli linguistici di grandi dimensioni (LLM) sono reti neurali addestrate su vasti corpus testuali, capaci di generare testo coerente, rispondere a domande e svolgere compiti linguistici complessi.",
        "La Computer Vision e una disciplina dell'AI che sviluppa tecniche per consentire ai computer di interpretare e comprendere i contenuti di immagini e video digitali.",
    ],
    "Healthcare AI": [
        "La diagnostica per immagini assistita da AI utilizza reti neurali convoluzionali per analizzare radiografie, TAC e risonanze magnetiche, supportando i radiologi nell'identificazione di patologie.",
        "Il drug discovery con intelligenza artificiale impiega modelli generativi e simulazioni molecolari per identificare candidati farmacologici promettenti, riducendo tempi e costi dello sviluppo.",
        "La telemedicina potenziata da AI integra dispositivi wearable e algoritmi predittivi per il monitoraggio remoto dei pazienti cronici, consentendo interventi tempestivi e personalizzati.",
        "La medicina personalizzata basata su genomica utilizza modelli di machine learning per analizzare il profilo genetico del paziente e ottimizzare la scelta del trattamento terapeutico.",
        "Il triage automatizzato con AI classifica la gravita dei pazienti in pronto soccorso analizzando sintomi, parametri vitali e anamnesi tramite modelli di NLP e machine learning.",
        "La patologia digitale utilizza scanner per digitalizzare i preparati istologici e algoritmi di deep learning per assistere i patologi nella diagnosi di tumori e altre patologie.",
        "L'analisi predittiva delle epidemie utilizza modelli epidemiologici potenziati da machine learning per prevedere la diffusione di malattie infettive e supportare le politiche sanitarie.",
    ],
    "Research": [
        "Le Physics-Informed Neural Networks (PINN) integrano le leggi fisiche nel processo di addestramento delle reti neurali per risolvere equazioni differenziali con applicazioni in fluidodinamica e meccanica.",
        "La scoperta di materiali con AI utilizza modelli generativi e simulazioni ab initio per esplorare lo spazio chimico e identificare nuovi materiali con proprieta desiderate.",
        "La bioinformatica computazionale applica reti neurali e modelli di linguaggio a sequenze proteiche e genomiche per predire strutture, funzioni e interazioni molecolari.",
        "Il calcolo ad alte prestazioni (HPC) potenziato da AI combina simulazioni su supercomputer con modelli di surrogato per accelerare le ricerche in fisica, climatologia e ingegneria.",
        "L'active learning per la ricerca scientifica ottimizza la progettazione degli esperimenti selezionando i campioni piu informativi da analizzare, riducendo costi e tempi sperimentali.",
        "La data-driven discovery utilizza tecniche di machine learning per identificare pattern nascosti in grandi dataset scientifici, portando a nuove scoperte in astronomia, genomica e scienza dei materiali.",
        "Le simulazioni Monte Carlo potenziate da reti neurali accelerano i calcoli stocastici in fisica delle particelle e finanza computazionale, mantenendo l'accuratezza statistica richiesta.",
    ],
    "Robotics and Industry": [
        "La robotica collaborativa (cobot) consente a robot e operatori umani di lavorare nello stesso spazio in sicurezza, con applicazioni nell'assemblaggio, nell'ispezione e nella movimentazione di materiali.",
        "Il digital twin industriale e una replica virtuale di un impianto produttivo che utilizza dati IoT in tempo reale per simulare, monitorare e ottimizzare i processi di produzione.",
        "La manutenzione predittiva Industry 4.0 analizza dati vibrazionali, termici e acustici dei macchinari con algoritmi di machine learning per prevedere guasti e pianificare interventi preventivi.",
        "La visione artificiale industriale utilizza telecamere e algoritmi di deep learning per il controllo qualita automatizzato, rilevando difetti dimensionali, estetici e strutturali in linea di produzione.",
        "La smart factory integra robotica, IoT, cloud computing e intelligenza artificiale per creare impianti produttivi autonomi, flessibili e capaci di auto-ottimizzarsi in tempo reale.",
        "Gli AMR (Autonomous Mobile Robots) navigano autonomamente nei magazzini e nelle fabbriche utilizzando SLAM e sensori per movimentare materiali senza infrastrutture fisse come binari o magneti.",
        "I sistemi cyber-fisici (CPS) integrano componenti computazionali con processi fisici industriali, consentendo il controllo in tempo reale e l'adattamento autonomo della produzione.",
    ],
    "Virtual assistants": [
        "I chatbot conversazionali basati su AI utilizzano modelli di comprensione del linguaggio naturale (NLU) per interpretare le richieste degli utenti e generare risposte pertinenti e contestuali.",
        "L'architettura RAG (Retrieval-Augmented Generation) combina la ricerca in basi documentali con modelli linguistici generativi per fornire risposte accurate e aggiornate alle domande degli utenti.",
        "Il dialogue management gestisce il flusso della conversazione tra utente e assistente virtuale, mantenendo il contesto attraverso piu turni e guidando l'interazione verso la risoluzione del problema.",
        "Il riconoscimento vocale (Speech-to-Text) trasforma il parlato in testo utilizzando modelli di deep learning, abilitando l'interazione vocale con assistenti virtuali in lingua italiana.",
        "La sintesi vocale (Text-to-Speech) converte il testo generato dall'assistente virtuale in parlato naturale, utilizzando modelli neurali per produrre voci realistiche ed espressive.",
        "L'intent recognition classifica l'intenzione dell'utente a partire dal testo della sua richiesta, utilizzando modelli di NLP per instradare la conversazione verso il flusso di dialogo appropriato.",
        "La memoria conversazionale a lungo termine consente agli assistenti virtuali di ricordare informazioni dalle conversazioni precedenti con lo stesso utente per personalizzare le interazioni future.",
    ],
}


def generate():
    total = 0
    for class_name, concepts in CLASS_CONCEPTS.items():
        slug = _slugify(class_name)
        out_dir = os.path.join(BASE_DATA_PATH, slug)
        os.makedirs(out_dir, exist_ok=True)

        for i, text in enumerate(concepts):
            filename = f"concept_{slug}_{i}.txt"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"  {class_name}: {len(concepts)} concepts in {slug}/")
        total += len(concepts)

    print(f"Total: {total} concept examples generated")


if __name__ == "__main__":
    generate()
