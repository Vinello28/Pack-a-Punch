import os
import uuid

ai_examples = [
    "Sviluppo di un sistema di visione artificiale basato su Deep Learning per il controllo qualità automatizzato nella linea di produzione.",
    "Implementazione di algoritmi di Machine Learning per la manutenzione predittiva dei macchinari industriali, riducendo i tempi di inattività.",
    "Creazione di un assistente virtuale intelligente basato su Natural Language Processing (NLP) per il supporto clienti automatizzato.",
    "Progetto di ricerca per l'ottimizzazione della logistica aziendale tramite reti neurali e algoritmi genetici.",
    "Piattaforma cloud integrata con modelli di AI generativa per la creazione automatizzata di contenuti di marketing.",
    "Applicazione di tecniche di Reinforcement Learning per la gestione autonoma e ottimizzata del consumo energetico negli edifici smart.",
    "Sviluppo di un sistema di raccomandazione e-commerce basato su intelligenza artificiale per personalizzare l'esperienza utente sulla piattaforma.",
    "Utilizzo della Computer Vision e reti neurali convoluzionali (CNN) per il monitoraggio e la sicurezza nei cantieri edili in tempo reale.",
    "Prototipazione di un software di analisi del sentiment basato su modelli linguistici di grandi dimensioni (LLM) per analizzare i feedback dei consumatori.",
    "Integrazione di modelli predittivi ML per l'analisi dei dati finanziari e la prevenzione automatizzata delle frodi, aumentando la sicurezza delle transazioni.",
    "Sviluppo di un sistema esperto basato su reti bayesiane per il supporto alle decisioni mediche e la diagnosi precoce di patologie rare.",
    "Implementazione di un'architettura di Edge AI per l'elaborazione dei dati dei sensori IoT direttamente sui macchinari industriali, riducendo la latenza.",
    "Realizzazione di un chatbot basato su intelligenza artificiale conversazionale per la pubblica amministrazione, al fine di semplificare l'accesso ai servizi.",
    "Utilizzo di algoritmi di clustering e rilevamento delle anomalie per la cybersecurity e la protezione avanzata delle reti aziendali contro attacchi zero-day.",
    "Progetto per l'automazione dei processi robotici (RPA) potenziata con capacità di apprendimento automatico e intelligenza artificiale cognitiva."
]

non_ai_examples = [
    "Rinnovamento degli impianti di illuminazione dello stabilimento produttivo con l'installazione di nuova tecnologia LED ad alta efficienza per ridurre i consumi.",
    "Acquisto di nuovi macchinari a controllo numerico (CNC) per aumentare la capacità produttiva del reparto fresatura e tornitura meccanica.",
    "Progetto di internazionalizzazione per la partecipazione a fiere di settore e l'apertura di un nuovo showroom nel mercato asiatico nei prossimi due anni.",
    "Implementazione di un nuovo sistema gestionale ERP per l'integrazione e la digitalizzazione dei processi contabili, amministrativi e logistici.",
    "Intervento di riqualificazione energetica dell'edificio aziendale mediante l'installazione di pannelli fotovoltaici e isolamento termico a cappotto.",
    "Sviluppo di una nuova piattaforma di e-commerce standard B2C per la vendita online dei prodotti aziendali, dotata di sistema di pagamento sicuro.",
    "Piano di formazione aziendale sulla sicurezza sul lavoro e l'aggiornamento normativo in ambito di privacy (GDPR) per tutto il personale dipendente.",
    "Adozione di un sistema di tracciabilità della filiera agroalimentare tramite tecnologia blockchain per certificare e garantire l'origine dei prodotti regionali.",
    "Acquisto di nuovi server e infrastrutture hardware per il potenziamento del data center aziendale e il passaggio della rete al cloud computing.",
    "Ristrutturazione e ampliamento dei locali aziendali per la creazione di nuovi uffici dirigenziali e un'area relax dedicata al benessere dei dipendenti.",
    "Sviluppo di un'applicazione mobile nativa per iOS e Android dedicata alla prenotazione dei servizi aziendali, con interfaccia grafica moderna (UI/UX).",
    "Campagna globale di web marketing, social media management e ottimizzazione SEO per incrementare la visibilità del brand sui motori di ricerca.",
    "Miglioramento del ciclo di confezionamento automatizzato mediante l'inserimento di nuovi nastri trasportatori, isole robotizzate e macchine confezionatrici.",
    "Consulenza specialistica per l'ottenimento della certificazione di qualità aziendale ISO 9001:2015 e l'ottimizzazione del relativo sistema di gestione qualità.",
    "Progettazione e realizzazione di un nuovo packaging ecosostenibile e totalmente biodegradabile per ridurre significativamente l'impatto ambientale dei nostri prodotti di punta."
]

ai_dir = "src/data/ai"
non_ai_dir = "src/data/non_ai"

os.makedirs(ai_dir, exist_ok=True)
os.makedirs(non_ai_dir, exist_ok=True)

def save_examples(examples, directory):
    count = 0
    for ex in examples:
        file_name = f"synthetic_project_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join(directory, file_name), "w", encoding="utf-8") as f:
            f.write(ex)
        count += 1
    print(f"Salvato {count} esempi sintetici in {directory}")

save_examples(ai_examples, ai_dir)
save_examples(non_ai_examples, non_ai_dir)
