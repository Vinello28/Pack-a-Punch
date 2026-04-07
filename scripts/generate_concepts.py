import os
import uuid

ai_definitions = [
    "Il Machine Learning è una branca dell'intelligenza artificiale che si occupa di creare sistemi in grado di apprendere dall'esperienza e dai dati, migliorando le loro prestazioni nel tempo senza istruzioni esplicite.",
    "Il Deep Learning è un sotto-campo del machine learning basato su reti neurali artificiali con strati multipli, capace di modellare astrazioni di alto livello nei dati, molto usato per video, audio e testo.",
    "L'elaborazione del linguaggio naturale (NLP) è un campo dell'intelligenza artificiale che studia le interazioni tra computer e linguaggio umano, permettendo alle macchine di comprendere e generare testo.",
    "La Computer Vision è una disciplina scientifica che cerca di sviluppare tecniche per aiutare i computer a 'vedere' e comprendere i contenuti di immagini digitali come fotografie o video in modo autonomo.",
    "Il Reinforcement Learning è una tecnica di machine learning in cui l'agente impara a prendere decisioni compiendo azioni in un ambiente per massimizzare una nozione di ricompensa cumulativa.",
    "Un modello generativo basato su IA, come i Large Language Models (LLM), è un sistema in grado di creare nuovi contenuti verosimili, come testi, codice o immagini, a partire da grandi moli di dati strutturati e non pre-esistenti.",
    "Le reti neurali convoluzionali (CNN) sono una classe speciale di reti neurali feed-forward, molto efficaci nel riconoscimento visivo, classificazione delle immagini e analisi di serie storiche.",
    "I sistemi di raccomandazione AI filtrano grandi quantità di informazioni per suggerire all'utente prodotti o contenuti rilevanti in base alle sue interazioni passate, apprendendone i gusti tramite reti neurali.",
    "Il clustering è una tecnica di apprendimento non supervisionato (Unsupervised Learning) che raggruppa un set di oggetti in modo tale che quelli nello stesso gruppo siano più simili tra loro che agli altri.",
    "Un sistema esperto è un programma informatico di intelligenza artificiale progettato per risolvere problemi complessi imitando le capacità decisionali di un esperto umano in un dominio specifico.",
    "L'Edge AI si riferisce all'esecuzione di algoritmi di intelligenza artificiale a livello di dispositivo fisico 'edge', come un sensore o uno smartphone, senza l'ausilio di calcolo remoto sul cloud per abbattere i tempi logistici.",
    "L'apprendimento supervisionato differisce da quello non supervisionato perché per addestrare l'algoritmo di machine learning vengono fornite coppie di dati in cui si ha l'input e l'output desiderato (etichettato per la classe).",
    "Gli alberi decisionali random forest sono dei modelli ad apprendimento automatico supervisionato che creano un insieme di 'decision tree' durante l'addestramento per outputtare la classe di maggioranza.",
    "Il riconoscimento vocale o speech-to-text si avvale dell'intelligenza artificiale per tradurre il linguaggio parlato in testo scritto, sfruttando le reti neurali ricorrenti e gli strati di attenzione profonda.",
    "La manutenzione predittiva tramite AI utilizza modelli matematici per analizzare lo storico dei guasti di macchinari per stimare in modo probabilistico in che momento sarà necessaria una futura manutenzione."
]

non_ai_definitions = [
    "La Blockchain è un registro pubblico decentralizzato noto come Distributed Ledger che utilizza algoritmi crittografici per mantenere le transazioni sicure, incorruttibili e trasparenti a tutti i partecipanti.",
    "L'Internet of Things (IoT) è costituito dall'insieme di dispositivi fisici dotati di sensori e connettività internet programmata, che scambiano e condividono dati per l'automazione locale senza intervento intelligente.",
    "Il Cloud Computing è un paradigma di distribuzione di risorse informatiche, come database e server di archiviazione, erogate tramite Internet e fornite on-demand agli utenti che ne fanno richiesta per risparmiare costo sugli apparati.",
    "I sistemi ERP o Enterprise Resource Planning sono software di gestione che integrano tutte le attività di rilevanza aziendale, come fatturazione, buste paga, vendite e controllo qualità, centralizzando i dati relazionali.",
    "La stampa 3D o manifattura additiva è un processo elettro-meccanico per la creazione di un oggetto solido tridimensionale partendo da un modello digitale, assemblando il materiale strato dopo strato base.",
    "Il fotovoltaico è una tecnologia per la produzione di energia elettrica sostenibile che sfrutta le particelle di luce solare per colpire delle superfici in silicio semiconduttore per generare un flusso continuo fotovoltaico.",
    "Il Search Engine Optimization (SEO) rappresenta l'insieme di pratiche tecniche e strategiche messe in campo per migliorare l'indicizzazione di un sito web nei risultati non a pagamento dei motori di ricerca informatici.",
    "Sviluppo di portali eCommerce B2B e B2C: questo processo riguarda la pura stesura di codice HTML, CSS e backend relazionale per costruire siti internet dedicati allo scambio di beni, servizi e carrelli acquisti utente.",
    "La realtà virtuale in ambienti di sviluppo è una simulazione immersiva di un ambiente tridimensionale creata in linguaggio macchina, con cui l'utente può interagire usando dispositivi elettronici speciali come visori o guanti aptici.",
    "Le macchine a Controllo Numerico (CNC) sono dispositivi industriali le cui operazioni e i cui movimenti di lavorazione sono precisamente comandati da un minicomputer o microcontrollore secondo schemi predefiniti di tornitura rigidi.",
    "Una rete 5G è uno standard tecnologico di quinta generazione per connessioni e comunicazioni mobili a banda larga, in grado di garantire velocità di picco maggiori di dati in gigabit per secondo rispetto al suo predecessore 4G.",
    "I database relazionali (RDBMS) organizzano le informazioni aziendali in tabelle di righe e colonne pre-codificate con chiavi di relazione, interrogabili tramite query scritte in linguaggio SQL per recuperare le informazioni storiche.",
    "La Cybersecurity aziendale comprende tutte le tecnologie IT (Firewall e Antivirus classici) atte a proteggere le infrastrutture informatiche e la rete aziendale mediante firme di virus note, senza analisi euristica complessa proattiva.",
    "I sistemi di Business Intelligence raccolgono dati grezzi da diverse fonti interne all'azienda, e tramite dashboard di data visualization, mostrano aggregati statistici, metriche storiche aggregate utili ai manager direzionali.",
    "L'isolamento termico in edilizia fa riferimento alle tecniche industriali usate in ambito immobiliare basate su materiali sintetici a bassa conducibilità come polistirene espanso termoisolante o lana di roccia, per far mantenere calore."
]

ai_dir = "src/data/ai"
non_ai_dir = "src/data/non_ai"

os.makedirs(ai_dir, exist_ok=True)
os.makedirs(non_ai_dir, exist_ok=True)

def save_examples(examples, directory, prefix):
    count = 0
    for ex in examples:
        file_name = f"{prefix}_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join(directory, file_name), "w", encoding="utf-8") as f:
            f.write(ex)
        count += 1
    print(f"Salvato {count} esempi concettuali in {directory}")

save_examples(ai_definitions, ai_dir, "concept_ai")
save_examples(non_ai_definitions, non_ai_dir, "concept_non_ai")
