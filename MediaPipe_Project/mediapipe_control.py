import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque

# ============================================================
# CONFIGURAZIONE AMBIENTE E PARAMETRI
# ============================================================
# Flag booleano per determinare la modalità operativa.
# True: Esegue il codice in modalità debug senza connessione fisica.
# False: Tenta la connessione reale tramite interfaccia radio/Wi-Fi al drone.
SIMULATION = True

if not SIMULATION:
    try:
        # Importazione condizionale della libreria pyparrot per la gestione del drone Mambo.
        from pyparrot.Minidrone import Mambo
    except ImportError:
        # Gestione dell'eccezione in caso di libreria mancante.
        print("ERRORE CRITICO: Libreria pyparrot non installata. Esegui 'pip install pyparrot'")
        exit()


# ============================================================
# CLASSE: GestureRecognizer
# ============================================================
class GestureRecognizer:
    """
    Classe responsabile dell'analisi geometrica dei landmarks della mano.
    Converte le coordinate spaziali (x, y, z) fornite da MediaPipe in
    classificazioni discrete di gesti (comandi).
    """

    def __init__(self):
        # Definizione degli indici dei keypoint secondo lo standard MediaPipe Hands.
        # TIP: Estremità del dito.
        # PIP: Articolazione interfalangea prossimale (snodo centrale).
        self.tip_indices = [4, 8, 12, 16, 20]  # Ordine: Pollice, Indice, Medio, Anulare, Mignolo
        self.pip_indices = [2, 6, 10, 14, 18]
        self.mcp_indices = [1, 5, 9, 13, 17]  # MCP: Articolazione della base delle mani

    def recognize_gesture(self, landmarks):
        """
        Analizza la configurazione della mano frame per frame.

        Args:
            landmarks: Lista di oggetti normalizzati contenenti coordinate x, y, z.

        Returns:
            Stringa rappresentante il comando rilevato ("UP", "DOWN", "LEFT", "RIGHT", "OTHER").
        """
        # Gestione del caso base: nessun landmark rilevato.
        if not landmarks:
            return "OTHER"

        # --- FASE 1: ANALISI STATO DITA (APERTO/CHIUSO) ---
        # In Computer Vision (OpenCV), l'origine (0,0) è l'angolo in alto a sinistra.
        # Pertanto, coordinata Y minore significa posizione fisica "in alto".
        fingers_extended = []
        for i in range(5):
            # Confronto posizionale: se la punta (TIP) è più in alto dello snodo (PIP),
            # il dito è considerato esteso.
            if landmarks[self.tip_indices[i]].y < landmarks[self.pip_indices[i]].y:
                fingers_extended.append(True)
            else:
                fingers_extended.append(False)

        # Mapping booleano per leggibilità semantica
        thumb_up = fingers_extended[0]
        index_up = fingers_extended[1]
        middle_up = fingers_extended[2]
        ring_up = fingers_extended[3]
        pinky_up = fingers_extended[4]

        # Logica di controllo: Medio, Anulare e Mignolo devono essere chiusi
        # per validare i gesti direzionali ed evitare falsi positivi.
        main_fingers_closed = not (middle_up or ring_up or pinky_up)

        # --- FASE 2: CLASSIFICAZIONE GESTI ---

        # CASO A: Gesti Direzionali (Richiede indice esteso e altre dita chiuse)
        if main_fingers_closed and index_up:
            # Estrazione coordinate per calcolo vettoriale
            index_tip = landmarks[8]  # Punta indice
            index_mcp = landmarks[5]  # Base indice

            # Calcolo delle componenti del vettore direzione
            dx = index_tip.x - index_mcp.x
            dy = index_tip.y - index_mcp.y

            # Calcolo dell'angolo tramite arcotangente (atan2 gestisce i quadranti correttamente).
            # Conversione da radianti a gradi.
            angle = np.degrees(np.arctan2(dy, dx))

            # Conversione dell'angolo in comandi:
            # Range [-45, +45]: Vettore orizzontale verso destra.
            if -45 <= angle <= 45:
                return "RIGHT"
            # Range [>135 o <-135]: Vettore orizzontale verso sinistra.
            elif angle >= 135 or angle <= -135:
                return "LEFT"
            # Range [-135, -45]: Vettore verticale verso l'alto (Y negativa).
            elif -135 < angle < -45:
                return "UP"

        # CASO B: Gesto DOWN (Logica biometrica per mano invertita/pollice verso)
        # Analisi relativa tra polso (Wrist) e base dell'indice (Index MCP).
        wrist_y = landmarks[0].y
        index_mcp_y = landmarks[5].y

        # Se il polso ha coordinata Y minore della base delle dita, significa che si trova
        # fisicamente più in alto. La mano è quindi orientata verso il basso.
        hand_inverted = wrist_y < index_mcp_y

        if hand_inverted and main_fingers_closed:
            return "DOWN"

        # Nessun pattern valido riconosciuto
        return "OTHER"


# ============================================================
# THREAD DI CONTROLLO
# ============================================================
def control_thread(mambo, lock, latest_command, is_flying, stop_event):
    """
    Thread dedicato all'invio asincrono dei comandi al drone.
    Gestisce la latenza di rete e implementa meccanismi di sicurezza (fail-safe).
    """
    # Timeout di sicurezza: se non arrivano nuovi dati per 0.5s, il drone si ferma.
    neutral_timeout = 0.5

    # Ciclo di vita del thread, attivo finché non viene settato l'evento di stop.
    while not stop_event.is_set():
        # Acquisizione del Lock per accesso thread-safe alla risorsa condivisa.
        with lock:
            cmd_data = latest_command[0]

        # Unpacking della tupla (comando, timestamp)
        command, timestamp = cmd_data

        # Verifica validità temporale del comando.
        # Se il comando è obsoleto, si forza uno stato nullo.
        if time.time() - timestamp > neutral_timeout:
            command = ""

        # Gestione prioritaria dell'atterraggio
        if command == "LAND":
            print("[CONTROL] SEQUENZA DI ATTERRAGGIO INIZIATA")
            if not SIMULATION:
                # Invocazione metodo di atterraggio sicuro SDK
                mambo.safe_land(timeout=10)
                mambo.smart_sleep(2)
            # Aggiornamento stato globale e interruzione loop
            is_flying[0] = False
            stop_event.set()
            break

        # Invio comando di volo attivo
        if is_flying[0] and command != "":
            if SIMULATION:
                # Feedback a console (sovrascrittura riga con \r per pulizia output)
                print(f"[SIM] Esecuzione attuatore: {command}   ", end='\r')
            else:
                # Mapping del comando su attuatori fisici (Roll, Pitch, Yaw, Vertical).
                # Argomenti: (roll, pitch, yaw, vertical_movement, duration)
                if command == "LEFT":
                    mambo.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                elif command == "RIGHT":
                    mambo.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
                elif command == "UP":
                    mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.1)
                elif command == "DOWN":
                    mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.1)

        # Sleep per limitare il polling rate e ridurre carico CPU (10Hz circa)
        time.sleep(0.1)

# ============================================================
# THREAD DI VISIONE E ELABORAZIONE
# ============================================================
def vision_thread(cap, hands, mp_drawing, recognizer, lock, latest_command, stop_event):
    """
    Thread dedicato all'acquisizione video e inferenza MediaPipe.
    Implementa la stabilizzazione del segnale tramite buffer FIFO.
    """
    # Coda a dimensione fissa per storicizzare gli ultimi 5 gesti (filtro temporale)
    gesture_history = deque(maxlen=5)

    def get_stable_gesture(dq):
        """
        Algoritmo di stabilizzazione:
        Restituisce un gesto valido solo se consistente per almeno 3 frame consecutivi.
        """
        if len(dq) < 3: return ""
        # Verifica frequenza dell'ultimo elemento nella coda
        if dq.count(dq[-1]) >= 3:
            return dq[-1]
        return ""

    print("[VISION] Sottosistema visione avviato.")

    while not stop_event.is_set():
        # Lettura frame dal buffer della webcam
        ret, frame = cap.read()
        if not ret: break  # Uscita in caso di errore hardware video

        # Flip orizzontale per effetto specchio
        frame = cv2.flip(frame, 1)
        # Conversione spazio colore da BGR (OpenCV default) a RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Esecuzione inferenza modello MediaPipe Hands
        results = hands.process(rgb_frame)

        current_gestures = []

        # Elaborazione landmarks se rilevati
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Rendering scheletro mano su frame originale
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                # Classificazione geometrica
                gesture = recognizer.recognize_gesture(hand_landmarks.landmark)
                current_gestures.append(gesture)

        # Logica di arbitraggio comandi (Priority Logic)
        final_cmd = ""
        # Se entrambe le mani indicano DOWN, comando prioritario LAND
        if current_gestures.count("DOWN") >= 2:
            final_cmd = "LAND"
        elif "LEFT" in current_gestures:
            final_cmd = "LEFT"
        elif "RIGHT" in current_gestures:
            final_cmd = "RIGHT"
        elif "UP" in current_gestures:
            final_cmd = "UP"
        elif "DOWN" in current_gestures:
            final_cmd = "DOWN"

        # Aggiornamento buffer storico
        gesture_history.append(final_cmd)
        # Calcolo del comando stabilizzato
        stable_cmd = get_stable_gesture(gesture_history)

        # Sezione Critica: Aggiornamento variabili condivise
        with lock:
            if stable_cmd:
                # Aggiornamento tupla (comando, timestamp corrente)
                latest_command[0] = (stable_cmd, time.time())

            # Lettura stato per aggiornamento dell'interfaccia utente
            current_executing = latest_command[0][0]
            # Verifica scadenza timeout per feedback visivo
            is_expired = (time.time() - latest_command[0][1]) > 0.5

        # Rendering interfaccia grafica
        # Verde se attivo, Rosso se in hover/timeout
        color = (0, 255, 0) if not is_expired and current_executing else (0, 0, 255)
        text_display = f"CMD: {current_executing}" if not is_expired else "CMD: HOVER"

        # Disegno background per testo
        cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, text_display, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Visualizzazione frame elaborato
        cv2.imshow("Drone Controller Logic", frame)

        # Gestione input tastiera per uscita (polling 1ms)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()  # Segnala a tutti i thread di terminare
            break

    # Rilascio risorse hardware
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================
def main():
    # Inizializzazione oggetto Drone
    mambo = None
    if not SIMULATION:
        print("Inizializzazione protocollo connessione Mambo...")
        # Istanza della classe Mambo (comunicazione Wi-Fi)
        mambo = Mambo(None, use_wifi=True)
        success = mambo.connect(num_retries=3)

        if not success:
            print("Errore handshake connessione. Verificare link Wi-Fi.")
            return

        print("Connessione stabilita. Esecuzione sequenza di decollo...")
        mambo.smart_sleep(1)  # Attesa stabilizzazione segnale
        mambo.safe_takeoff(5)  # Decollo con timeout 5s
        mambo.smart_sleep(1)

    # Stato di volo condiviso (lista mutabile per riferimento)
    is_flying = [True]

    # Configurazione Pipeline MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,  # Massimo numero di mani tracciabili
        min_detection_confidence=0.7,  # Soglia confidenza rilevamento iniziale
        min_tracking_confidence=0.5  # Soglia confidenza tracking continuo
    )
    mp_drawing = mp.solutions.drawing_utils
    recognizer = GestureRecognizer()

    # Primitive di sincronizzazione Thread
    lock = threading.Lock()  # Mutex per accesso esclusivo ai dati
    latest_command = [("", 0)]  # Struttura dati condivisa: [Comando, Timestamp]
    stop_event = threading.Event()  # Flag per terminazione controllata

    # Istanziazione dei Thread
    # Thread Visione: gestisce input sensore (camera)
    t_vision = threading.Thread(
        target=vision_thread,
        args=(cv2.VideoCapture(0), hands, mp_drawing, recognizer, lock, latest_command, stop_event)
    )
    # Thread Controllo: gestisce output attuatore (drone)
    t_control = threading.Thread(
        target=control_thread,
        args=(mambo, lock, latest_command, is_flying, stop_event)
    )

    # Avvio esecuzione concorrente
    t_vision.start()
    t_control.start()

    # Sincronizzazione finale (attesa terminazione thread)
    t_vision.join()
    t_control.join()

    # Clean-up finale
    if not SIMULATION and mambo:
        print("Chiusura connessione drone...")
        mambo.disconnect()


# Idioma Python per esecuzione script
if __name__ == "__main__":
    main()
