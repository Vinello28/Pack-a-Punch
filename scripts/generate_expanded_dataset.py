import os
import random

ai_dir = "/home/gabs/Documenti/University/Organizzazione dell'Impresa/Pack-a-Punch/src/data/ai"
non_ai_dir = "/home/gabs/Documenti/University/Organizzazione dell'Impresa/Pack-a-Punch/src/data/non_ai"

# Template components for generating diverse AI project descriptions
ai_domains = [
    "sanità", "agricoltura", "manifatturiero", "finanza", "trasporti", "energia",
    "telecomunicazioni", "retail", "logistica", "sicurezza", "ambiente", "istruzione",
    "turismo", "pubblica amministrazione", "ricerca scientifica", "automotive", "aerospace",
    "farmaceutico", "alimentare", "chimico", "edilizia", "moda", "media", "sport"
]

ai_techniques = [
    ("Deep Learning con reti neurali convoluzionali (CNN)", "CNN"),
    ("Recurrent Neural Networks (RNN) e varianti LSTM/GRU", "RNN_LSTM"),
    ("architetture Transformer e modelli di attenzione", "Transformer"),
    ("Reinforcement Learning (RL) con algoritmi PPO/DQN", "RL"),
    ("Graph Neural Networks (GNN) per dati strutturati", "GNN"),
    ("Generative Adversarial Networks (GAN)", "GAN"),
    ("Variational Autoencoders (VAE)", "VAE"),
    ("Transfer Learning e fine-tuning di modelli pre-addestrati", "TransferLearning"),
    ("Few-Shot Learning e Meta-Learning", "FewShot"),
    ("Self-Supervised Learning (SSL)", "SSL"),
    ("Federated Learning per privacy-preserving AI", "FederatedLearning"),
    ("Neural Architecture Search (NAS)", "NAS"),
    ("Explainable AI (XAI) con SHAP/LIME", "XAI"),
    ("Active Learning per ottimizzazione etichettatura", "ActiveLearning"),
    ("Ensemble Learning con Random Forest e Gradient Boosting", "Ensemble"),
    ("Support Vector Machines (SVM) con kernel non lineari", "SVM"),
    ("K-Nearest Neighbors (KNN) e algoritmi di clustering", "KNN"),
    ("algoritmi di Anomaly Detection", "AnomalyDetection"),
    ("Natural Language Processing (NLP) con BERT/GPT", "NLP"),
    ("Computer Vision e Object Detection (YOLO/R-CNN)", "ComputerVision"),
    ("Semantic Segmentation e Instance Segmentation", "Segmentation"),
    ("Speech Recognition con modelli end-to-end", "SpeechRecognition"),
    ("Time Series Forecasting con modelli ARIMA-LSTM ibridi", "TimeSeries"),
    ("Recommendation Systems con Collaborative Filtering", "RecommenderSystems"),
    ("Reinforcement Learning from Human Feedback (RLHF)", "RLHF"),
    ("Multi-Agent Reinforcement Learning", "MARL"),
    ("Continual Learning e Lifelong Learning", "ContinualLearning"),
    ("Edge AI e TinyML per dispositivi embedded", "EdgeAI"),
    ("Quantum Machine Learning", "QuantumML"),
    ("Neuromorphic Computing", "Neuromorphic")
]

ai_applications = [
    "diagnosi medica automatizzata", "manutenzione predittiva", "controllo qualità",
    "ottimizzazione processi produttivi", "fraud detection", "credit scoring",
    "sentiment analysis", "chatbot e assistenti virtuali", "traduzione automatica",
    "riconoscimento facciale", "object tracking", "guida autonoma", "pianificazione percorsi",
    "previsione domanda", "ottimizzazione inventario", "pricing dinamico",
    "rilevamento intrusioni", "analisi video sorveglianza", "monitoraggio ambientale",
    "previsione meteo", "analisi immagini satellitari", "drug discovery",
    "protein folding", "genomica personalizzata", "riconoscimento vocale",
    "sintesi vocale", "generazione testo", "summarization", "question answering",
    "image generation", "video synthesis", "music generation", "style transfer",
    "super-resolution", "denoising", "inpainting", "colorization",
    "pose estimation", "action recognition", "scene understanding",
    "3D reconstruction", "SLAM", "path planning", "grasp planning",
    "human-robot interaction", "swarm intelligence", "traffic optimization",
    "energy management", "smart grid", "predictive maintenance",
    "churn prediction", "customer segmentation", "lead scoring",
    "content recommendation", "personalization", "A/B testing optimization"
]

ai_data_sources = [
    "Big Data provenienti da sensori IoT distribuiti",
    "dataset annotati di oltre 1 milione di esempi",
    "immagini ad alta risoluzione acquisite da telecamere industriali",
    "serie temporali storiche di 10+ anni",
    "dati multimodali (testo, immagini, audio)",
    "stream di dati in tempo reale",
    "database relazionali legacy migrati su data lake",
    "log di sistema e telemetria applicativa",
    "dati geospaziali e immagini satellitari multispettrali",
    "segnali biomedici (EEG, ECG, fMRI)",
    "dati finanziari tick-by-tick ad alta frequenza",
    "corpus testuali multilingue",
    "video ad alta definizione 4K/8K",
    "point cloud da sensori LiDAR",
    "dati sintetici generati tramite simulazione"
]

ai_infrastructure = [
    "cluster GPU NVIDIA A100/H100 per training distribuito",
    "infrastruttura cloud scalabile (AWS/Azure/GCP)",
    "edge computing con acceleratori TPU/NPU",
    "architettura serverless per inferenza",
    "container orchestration con Kubernetes",
    "pipeline MLOps con CI/CD automatizzato",
    "data warehouse su Snowflake/BigQuery",
    "feature store centralizzato",
    "model registry e versioning",
    "monitoring e observability con Prometheus/Grafana",
    "A/B testing framework",
    "distributed training con Horovod/DeepSpeed",
    "mixed precision training (FP16/BF16)",
    "model compression e quantization",
    "ONNX runtime per deployment cross-platform"
]

# Generate 500 AI project descriptions
ai_projects = []
for i in range(500):
    domain = random.choice(ai_domains)
    technique, tech_code = random.choice(ai_techniques)
    application = random.choice(ai_applications)
    data_source = random.choice(ai_data_sources)
    infra = random.choice(ai_infrastructure)
    
    templates = [
        f"Progetto di {application} nel settore {domain} basato su {technique}. Il sistema analizza {data_source} per ottimizzare i processi operativi. L'infrastruttura prevede {infra}. Il finanziamento copre sviluppo software, acquisizione dati e deployment in produzione.",
        
        f"Implementazione di un sistema di intelligenza artificiale per {application} utilizzando {technique}. Il progetto si rivolge al settore {domain} e processa {data_source}. L'architettura tecnologica include {infra} con capacità di scaling automatico.",
        
        f"Sviluppo di una piattaforma AI per {application} tramite {technique} applicata al dominio {domain}. Il modello viene addestrato su {data_source} e deployato su {infra}. I fondi supportano ricerca, sviluppo e validazione sul campo.",
        
        f"Realizzazione di un sistema di Machine Learning per {application} nel contesto {domain}. La soluzione utilizza {technique} per analizzare {data_source} in tempo reale. L'infrastruttura tecnologica prevede {infra} per garantire alte performance.",
        
        f"Progetto di ricerca applicata su {technique} per {application} in ambito {domain}. Il sistema elabora {data_source} attraverso pipeline di data processing avanzate. Il deployment avviene su {infra} con monitoraggio continuo delle performance.",
        
        f"Creazione di un motore di AI per {application} basato su {technique} dedicato al settore {domain}. L'addestramento sfrutta {data_source} con tecniche di data augmentation. L'infrastruttura include {infra} per training e inferenza.",
        
        f"Implementazione di algoritmi di {technique} per {application} nel dominio {domain}. Il progetto integra {data_source} provenienti da multiple sorgenti eterogenee. Il finanziamento copre {infra} e team di data scientist specializzati.",
        
        f"Sviluppo di un sistema intelligente per {application} tramite {technique} applicato al settore {domain}. La piattaforma processa {data_source} con latenza inferiore a 100ms. L'architettura prevede {infra} con ridondanza geografica.",
        
        f"Progetto di innovazione digitale per {application} basato su {technique} nel contesto {domain}. Il modello apprende da {data_source} utilizzando tecniche di curriculum learning. Il deployment sfrutta {infra} per garantire affidabilità.",
        
        f"Realizzazione di una soluzione AI per {application} mediante {technique} dedicata al mercato {domain}. Il sistema analizza {data_source} con accuratezza superiore al 95%. L'infrastruttura tecnologica comprende {infra} e sistemi di backup automatizzati."
    ]
    
    description = random.choice(templates)
    filename = f"ai_project_{i:04d}_{tech_code}"
    ai_projects.append((description, filename))

# Template components for Non-AI projects
nonai_sectors = [
    "infrastrutture stradali", "edilizia pubblica", "patrimonio culturale", "ambiente",
    "sociale", "turismo", "commercio", "artigianato", "agricoltura tradizionale",
    "pesca", "forestale", "idrico", "energetico tradizionale", "trasporti pubblici",
    "sanità", "istruzione", "sport", "cultura", "sicurezza urbana", "protezione civile"
]

nonai_interventions = [
    ("ristrutturazione", "renovation"), ("costruzione", "construction"),
    ("ampliamento", "expansion"), ("manutenzione straordinaria", "maintenance"),
    ("restauro conservativo", "restoration"), ("riqualificazione", "redevelopment"),
    ("bonifica", "remediation"), ("messa in sicurezza", "safety"),
    ("adeguamento normativo", "compliance"), ("efficientamento energetico", "energy_efficiency"),
    ("digitalizzazione", "digitalization"), ("ammodernamento", "modernization"),
    ("potenziamento", "enhancement"), ("realizzazione", "realization"),
    ("installazione", "installation"), ("sostituzione", "replacement"),
    ("consolidamento", "consolidation"), ("impermeabilizzazione", "waterproofing"),
    ("rifacimento", "refurbishment"), ("conversione", "conversion")
]

nonai_assets = [
    "edifici scolastici", "strutture ospedaliere", "impianti sportivi", "musei",
    "teatri", "biblioteche", "strade urbane", "ponti", "viadotti", "gallerie",
    "reti fognarie", "acquedotti", "impianti di depurazione", "parchi pubblici",
    "aree verdi", "piste ciclabili", "parcheggi", "illuminazione pubblica",
    "segnaletica stradale", "arredi urbani", "fontane", "monumenti storici",
    "chiese", "palazzi storici", "ville", "castelli", "torri", "mura cittadine",
    "impianti fotovoltaici", "caldaie", "ascensori", "impianti elettrici",
    "impianti idraulici", "coperture", "facciate", "infissi", "pavimentazioni"
]

nonai_technologies = [
    "materiali eco-compatibili certificati", "tecniche costruttive tradizionali",
    "isolamento termico ad alte prestazioni", "serramenti a taglio termico",
    "pannelli solari termici", "pompe di calore ad alta efficienza",
    "caldaie a condensazione", "sistemi di ventilazione meccanica controllata",
    "illuminazione LED di ultima generazione", "sistemi di raccolta acque piovane",
    "pavimentazioni drenanti", "barriere acustiche fonoassorbenti",
    "geotessili per consolidamento terreni", "reti paramassi certificate",
    "pali di fondazione trivellati", "micropali iniettati",
    "solette collaboranti in acciaio-calcestruzzo", "travi reticolari prefabbricate",
    "pannelli sandwich coibentati", "membrane impermeabilizzanti bituminose",
    "intonaci deumidificanti traspiranti", "pitture antimuffa",
    "sistemi antincendio sprinkler", "porte tagliafuoco certificate REI",
    "vetri stratificati di sicurezza", "inferriate di protezione",
    "sistemi di allarme perimetrale", "telecamere IP Full HD",
    "tornelli di accesso controllato", "badge RFID per controllo accessi"
]

nonai_objectives = [
    "migliorare la sicurezza degli utenti", "ridurre i consumi energetici",
    "adeguare alle normative vigenti", "aumentare l'accessibilità per disabili",
    "preservare il patrimonio storico", "valorizzare il territorio",
    "promuovere la mobilità sostenibile", "contrastare il dissesto idrogeologico",
    "ridurre l'inquinamento ambientale", "favorire l'inclusione sociale",
    "supportare le attività economiche locali", "migliorare la qualità della vita",
    "potenziare i servizi ai cittadini", "tutelare la salute pubblica",
    "preservare la biodiversità", "promuovere la cultura",
    "incentivare il turismo", "creare nuovi posti di lavoro",
    "ridurre le barriere architettoniche", "modernizzare le infrastrutture"
]

nonai_funding_details = [
    "Il finanziamento copre progettazione esecutiva, direzione lavori e collaudo finale",
    "I fondi supportano l'acquisto dei materiali, la manodopera specializzata e le opere accessorie",
    "Il progetto prevede il coinvolgimento di imprese certificate e l'utilizzo di materiali a norma",
    "Il budget include gli oneri di sicurezza, le spese tecniche e gli imprevisti",
    "Il finanziamento è destinato alle opere civili, agli impianti e alle finiture",
    "I fondi coprono le indagini preliminari, i rilievi topografici e le analisi strutturali",
    "Il progetto include la bonifica amianto, lo smaltimento rifiuti e il ripristino ambientale",
    "Il finanziamento supporta l'iter autorizzativo, le certificazioni e i collaudi",
    "I fondi sono destinati all'acquisto di attrezzature, macchinari e mezzi d'opera",
    "Il budget prevede la formazione del personale e l'assistenza tecnica post-operam"
]

# Generate 500 Non-AI project descriptions  
non_ai_projects = []
for i in range(500):
    sector = random.choice(nonai_sectors)
    intervention, interv_code = random.choice(nonai_interventions)
    asset = random.choice(nonai_assets)
    technology = random.choice(nonai_technologies)
    objective = random.choice(nonai_objectives)
    funding = random.choice(nonai_funding_details)
    
    templates = [
        f"Progetto di {intervention} di {asset} nel settore {sector}. L'intervento prevede l'utilizzo di {technology} per {objective}. {funding}.",
        
        f"Intervento di {intervention} finalizzato a {objective} attraverso il miglioramento di {asset}. Il progetto adotta {technology} conformi alle normative europee. {funding}.",
        
        f"Realizzazione di opere di {intervention} su {asset} esistenti per {objective}. Le tecnologie impiegate includono {technology}. {funding}.",
        
        f"Programma di {intervention} nel settore {sector} con focus su {asset}. L'intervento utilizza {technology} certificate. {funding}.",
        
        f"Piano di {intervention} di {asset} per {objective}. Il progetto prevede l'impiego di {technology} e il rispetto dei tempi contrattuali. {funding}.",
        
        f"Iniziativa di {intervention} dedicata a {asset} nel contesto {sector}. Le opere includono {technology} di ultima generazione. {funding}.",
        
        f"Progetto integrato di {intervention} per {objective} tramite interventi su {asset}. Le soluzioni tecniche adottate comprendono {technology}. {funding}.",
        
        f"Intervento straordinario di {intervention} su {asset} finalizzato a {objective}. Il cantiere utilizzerà {technology} con certificazione di qualità. {funding}.",
        
        f"Opera pubblica di {intervention} nel settore {sector} che interessa {asset}. Il progetto implementa {technology} nel rispetto dei criteri ambientali minimi. {funding}.",
        
        f"Lavori di {intervention} per {objective} attraverso il rinnovamento di {asset}. L'appalto prevede l'uso di {technology} e garanzie decennali. {funding}."
    ]
    
    description = random.choice(templates)
    filename = f"nonai_project_{i:04d}_{interv_code}"
    non_ai_projects.append((description, filename))

def generate_files(projects, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for text, filename in projects:
        file_path = os.path.join(base_dir, f"{filename}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

generate_files(ai_projects, ai_dir)
generate_files(non_ai_projects, non_ai_dir)
print(f"✓ Generati {len(ai_projects)} file AI")
print(f"✓ Generati {len(non_ai_projects)} file Non-AI")
print(f"✓ Totale dataset: {len(ai_projects) + len(non_ai_projects)} esempi")
