import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque
from pyparrot.Minidrone import Mambo

# ============================================================
# CLASSE DI RICONOSCIMENTO GESTI
# ============================================================
class GestureRecognizer:
    """
    Riconosce i gesti principali di una singola mano
    usando i punti chiave (landmarks) di MediaPipe.
    """
    def __init__(self):
        # Indici dei landmark delle punte e articolazioni PIP delle dita
        self.tip_indices = [4, 8, 12, 16, 20]  # Pollice, Indice, Medio, Anulare, Mignolo
        self.pip_indices = [2, 6, 10, 14, 18]

    def recognize_gesture(self, landmarks):
        """
        Ritorna la gesture rilevata per una mano singola:
        LEFT / RIGHT / UP / DOWN / LAND / OTHER
        """
        if landmarks is None:
            return "OTHER"

        # Controlla se medio, anulare e mignolo sono chiusi
        middle_closed = landmarks[self.tip_indices[2]].y > landmarks[self.pip_indices[2]].y
        ring_closed   = landmarks[self.tip_indices[3]].y > landmarks[self.pip_indices[3]].y
        pinky_closed  = landmarks[self.tip_indices[4]].y > landmarks[self.pip_indices[4]].y
        main_fingers_closed = middle_closed and ring_closed and pinky_closed

        # Controlla se l'indice è alzato
        index_extended = landmarks[self.tip_indices[1]].y < landmarks[self.pip_indices[1]].y

        # Controlla se il pollice è abbassato
        thumb_down = landmarks[self.tip_indices[0]].y > landmarks[self.pip_indices[0]].y

        # --- LEFT / RIGHT ---
        # Richiede indice alzato e altre dita chiuse
        if main_fingers_closed and index_extended:
            index_tip = landmarks[self.tip_indices[1]]
            index_mcp = landmarks[5]  # Base dell'indice
            dx = index_tip.x - index_mcp.x
            dy = index_tip.y - index_mcp.y
            angle = np.arctan2(dy, dx) * 180 / np.pi

            if -45 <= angle <= 45:
                return "RIGHT"  # Indice orizzontale a destra
            elif angle >= 135 or angle <= -135:
                return "LEFT"   # Indice orizzontale a sinistra

        # --- UP ---
        # Funziona anche con altre dita aperte, basta indice alzato
        if index_extended:
            return "UP"

        # --- DOWN / LAND ---
        # Se pollice abbassato e dita principali chiuse + indice non alzato -> DOWN
        if thumb_down and main_fingers_closed and not index_extended:
            return "DOWN"

        return "OTHER"  # Nessun gesto riconosciuto

# ============================================================
# THREAD DI CONTROLLO DEL DRONE
# ============================================================
def control_thread(mambo, lock, latest_command, is_flying):
    """
    Thread separato per inviare i comandi al drone.
    Legge il comando stabilizzato e lo esegue.
    """
    neutral_timeout = 0.5  # Se non riceve comando in 0.5s, comando neutro
    while True:
        with lock:
            command, timestamp = latest_command[0]

        # Azzeramento comando se vecchio
        if time.time() - timestamp > neutral_timeout:
            command = ""

        # Comandi al drone
        if command == "LEFT":
            mambo.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.2)
        elif command == "RIGHT":
            mambo.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.2)
        elif command == "UP":
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=0.2)
        elif command == "DOWN":
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-30, duration=0.2)
        elif command == "LAND":
            print("Comando LAND ricevuto. Atterraggio sicuro...")
            mambo.safe_land(timeout=10)
            is_flying[0] = False
            break  # Termina il thread dopo l'atterraggio

        time.sleep(0.05)  # Piccola pausa per non sovraccaricare il drone

# ============================================================
# THREAD DI VISIONE (GESTURE + VIDEO)
# ============================================================
def vision_thread(cap, hands, mp_drawing, recognizer, lock, latest_command):
    """
    Thread per acquisire video e riconoscere le gesture.
    Aggiorna il comando stabilizzato condiviso con il thread di controllo.
    """
    gesture_history = deque(maxlen=5)  # Memorizza ultime 5 gesture

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Effetto specchio
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        all_gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Riconosce la gesture di ogni mano
                gesture = recognizer.recognize_gesture(hand_landmarks.landmark)
                all_gestures.append(gesture)
                # Disegna la mano sul video
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # --- Logica dei comandi ---
        final_command = ""
        if all_gestures.count("DOWN") == 2:
            final_command = "LAND"  # Atterraggio sicurezza con entrambi i pollici
        elif "LEFT" in all_gestures:
            final_command = "LEFT"
        elif "RIGHT" in all_gestures:
            final_command = "RIGHT"
        elif all_gestures.count("UP") >= 1:
            final_command = "UP"   # Salita con un indice
        elif all_gestures.count("DOWN") == 1:
            final_command = "DOWN" # Discesa con un pollice solo

        # Stabilizzazione: deve comparire almeno 3 volte
        gesture_history.append(final_command)
        if gesture_history.count(final_command) >= 3:
            with lock:
                latest_command[0] = (final_command, time.time())
        else:
            with lock:
                latest_command[0] = ("", time.time())

        # Mostra il comando sul video
        cv2.putText(frame, latest_command[0][0], (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Drone Gesture Control", frame)

        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================
def main():
    mambo_address = None  # Indirizzo Wi-Fi del drone
    print("Connessione al Mambo...")
    mambo = Mambo(mambo_address, use_wifi=True)
    success = mambo.connect(num_retries=5)
    print(f"Connessione riuscita: {success}")
    if not success:
        print("Connessione fallita. Uscita.")
        return

    is_flying = [False]  # Stato iniziale: motori spenti

    # Aggiorna sensori e decollo automatico
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    mambo.safe_takeoff(timeout=10)  # Decollo appena parte il programma
    is_flying[0] = True

    # Setup webcam e MediaPipe
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    recognizer = GestureRecognizer()

    # Variabile condivisa tra thread
    lock = threading.Lock()
    latest_command = [("", time.time())]  # Comando iniziale vuoto

    # Avvio dei thread
    t_control = threading.Thread(target=control_thread, args=(mambo, lock, latest_command, is_flying))
    t_vision  = threading.Thread(target=vision_thread, args=(cap, hands, mp_drawing, recognizer, lock, latest_command))
    t_control.start()
    t_vision.start()
    t_vision.join()  # Aspetta che il thread video termini

    # Pulizia finale
    cap.release()
    cv2.destroyAllWindows()
    if is_flying[0] and not mambo.is_landed():
        mambo.safe_land(timeout=10)
    mambo.disconnect()
    print("Chiusura completata.")

if __name__ == "__main__":
    main()

