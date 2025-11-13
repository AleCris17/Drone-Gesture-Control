# Drone Gesture Control using MediaPipe

Un sistema di controllo gestuale per il drone Parrot Mambo basato su Computer Vision. Il progetto utilizza Python, OpenCV e MediaPipe per riconoscere i gesti della mano e tradurli in comandi di volo in tempo reale.

## Descrizione
Questo software permette di pilotare un drone senza l'ausilio di un controller fisico, utilizzando esclusivamente i movimenti della mano catturati da una webcam. L'algoritmo rileva i landmarks della mano, interpreta la configurazione geometrica delle dita e invia i comandi corrispondenti al drone tramite connessione wireless.

Il sistema è progettato su un'architettura Multithreading per disaccoppiare l'elaborazione visiva (che richiede fluidità) dall'invio dei comandi (soggetto a latenze di rete), garantendo un controllo stabile e reattivo.

## Funzionalità
* **Riconoscimento Real-Time:** Tracking della mano a bassa latenza.
* **Comandi di Volo:**
    * UP: Indice esteso verso l'alto.
    * DOWN: Pollice verso il basso (riduzione altitudine).
    * LEFT: Indice puntato verso sinistra.
    * RIGHT: Indice puntato verso destra.
    * LAND: Atterraggio di emergenza (attivato mostrando il gesto DOWN con entrambe le mani).
* **Modalità Simulazione:** Possibilità di testare la logica di riconoscimento e il feedback visivo senza connettere il drone fisico.
* **Stabilizzazione del Segnale:** Implementazione di un filtro temporale (deque) per evitare l'invio di comandi errati dovuti al tremolio della mano o incertezze del riconoscimento.

## Requisiti Tecnici
* **Linguaggio:** Python 3.12 o inferiore
* **Librerie:**
    * opencv-python (Elaborazione immagini)
    * mediapipe (Riconoscimento scheletro mano)
    * numpy (Calcoli matematici e geometrici)
    * pyparrot (Interfaccia di comunicazione con il drone Parrot Mambo)

## Installazione

1. Clonare la repository locale:
   git clone https://github.com/AleCris17/Drone-Gesture-Control.git

2. Installare le dipendenze necessarie tramite pip:
   pip install requirements.tex

## Utilizzo

1. Assicurarsi che la webcam sia collegata e funzionante.
2. Configurazione della modalità:
   * Aprire il file principale dello script.
   * Impostare la variabile `SIMULATION = True` per testare il riconoscimento a video.
   * Impostare la variabile `SIMULATION = False` per connettersi al drone reale (richiede connessione Bluetooth/Wi-Fi attiva con il Parrot Mambo).
3. Eseguire lo script:
   python main.py
4. Per terminare l'esecuzione e atterrare (se in volo), premere il tasto 'q'.

## Analisi Tecnica: MediaPipe vs YOLO

Durante la fase di sviluppo è stato valutato l'utilizzo di reti neurali per Object Detection (come YOLO). Tuttavia, la scelta finale è ricaduta su MediaPipe per i seguenti motivi ingegneristici:

1. **Efficienza Computazionale:** MediaPipe è altamente ottimizzato per l'esecuzione su CPU. Questo permette di ottenere un frame-rate elevato anche su hardware non dotato di GPU dedicata, condizione necessaria per un controllo fluido del drone.
2. **Landmarks 3D vs Bounding Box:** A differenza di YOLO, che restituisce un riquadro attorno all'oggetto, MediaPipe fornisce le coordinate spaziali (x, y, z) di 21 punti articolari della mano.
3. **Approccio Deterministico:** L'uso dei landmarks permette di calcolare angoli e distanze esatte tra le dita. Questo consente di definire i gesti tramite regole geometriche precise (es. calcolo dell'arcotangente per l'inclinazione dell'indice), offrendo una granularità di controllo superiore rispetto alla semplice classificazione di immagini.

## Struttura del Codice

* **GestureRecognizer:** Classe responsabile dell'analisi geometrica dei landmarks. Contiene la logica per determinare se le dita sono aperte o chiuse e calcolare l'orientamento della mano.
* **Vision Thread:** Thread dedicato all'acquisizione dei frame dalla webcam, all'elaborazione MediaPipe e all'aggiornamento dell'interfaccia grafica.
* **Control Thread:** Thread separato che legge il comando più recente, applica logiche di timeout di sicurezza e invia l'istruzione al drone.

## Autore: Alessandro Pio Crisetti
Progetto sviluppato per il tirocinio curriculare.
