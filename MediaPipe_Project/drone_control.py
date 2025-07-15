import cv2
import mediapipe as mp
import numpy as np


# --- CLASSE DI RICONOSCIMENTO GESTI ---
class GestureRecognizer:
    def __init__(self):
        # Salvare i punti chiave per riconoscere le gesture
        self.tip_indices = [4, 8, 12, 16, 20] # Punte di tutte le dita
        self.pip_indices = [2, 6, 10, 14, 18] # Articolazioni intermedie

    def recognize_gesture(self, landmarks):
        """
        Riconosce il gesto 'atomico' di una singola mano.
        """
        if landmarks is None: return "OTHER"

        # Controlla se le 3 dita principali (Medio, Anulare, Mignolo) sono chiuse, comparando le distanze
        middle_finger_closed = landmarks[self.tip_indices[2]].y > landmarks[self.pip_indices[2]].y
        ring_finger_closed = landmarks[self.tip_indices[3]].y > landmarks[self.pip_indices[3]].y
        pinky_finger_closed = landmarks[self.tip_indices[4]].y > landmarks[self.pip_indices[4]].y

        # Variabile che ci dice se tutte e tre le dita sono chiuse
        main_fingers_closed = middle_finger_closed and ring_finger_closed and pinky_finger_closed

        # Controlla se l'indice è esteso
        index_finger_extended = landmarks[self.tip_indices[1]].y < landmarks[self.pip_indices[1]].y

        # GESTO DI PUNTAMENTO (qualsiasi mano)
        if main_fingers_closed and index_finger_extended:
            index_tip = landmarks[self.tip_indices[1]]
            index_mcp = landmarks[5]

            dx = index_tip.x - index_mcp.x
            dy = index_tip.y - index_mcp.y

            # Movimento prevalentemente orizzontale
            if abs(dx) > abs(dy):
                # --- LOGICA INVERTITA ---
                # Se il dito punta a SINISTRA sullo schermo -> comando SINISTRA
                if dx < 0:
                    return "GO_LEFT"
                # Se il dito punta a DESTRA sullo schermo -> comando DESTRA
                else:
                    return "GO_RIGHT"
            # Movimento prevalentemente verticale
            elif abs(dy) > abs(dx):
                if dy < 0: return "INDEX_UP"

        # GESTO POLLICE IN GIU'
        index_finger_closed = not index_finger_extended
        thumb_is_down = landmarks[self.tip_indices[0]].y > landmarks[self.pip_indices[0]].y
        if main_fingers_closed and index_finger_closed and thumb_is_down:
            return "THUMB_DOWN"

        return "OTHER"


# --- SCRIPT PRINCIPALE ---
if __name__ == '__main__':
    # Var per selezionare una webcam
    cap = cv2.VideoCapture(0)

    # Usiamo la soluzione hand traking da mediapipe
    mp_hands = mp.solutions.hands # Riconoscimento delle mani
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) # Estraiamo il metodo (massimo due mani con la precisione al 50%)
    mp_drawing = mp.solutions.drawing_utils # Strumenti per tracciare lo scheletro della mano
    recognizer = GestureRecognizer() # Strumenti per riconoscere le gesture

    while cap.isOpened():
        # Salva il frame della webcam
        ret, frame = cap.read()
        if not ret: break

        # Flippa, converte da BGR a RGB e processa l'immagine con la var hands (quindi con il metodo Hands() di mp)
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        final_command = ""
        # Lista per salvare i gesti di tutte le mani rilevate
        all_gestures = []


        # Se viene identificata almeno una mano
        if results.multi_hand_landmarks:
            # Ciclo che viene fatto per ogni mano
            for hand_landmarks in results.multi_hand_landmarks:
                # Passa i dati al metodo
                gesture = recognizer.recognize_gesture(hand_landmarks.landmark)
                # Salava la serie di gesture nella lista
                all_gestures.append(gesture)
                # Disegna lo scheletro della mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Logica di Comando Finale ---
            # Comando laterale: ha la priorita' e basta una sola mano
            if "GO_LEFT" in all_gestures:
                final_command = "Vai a Sinistra"
            elif "GO_RIGHT" in all_gestures:
                final_command = "Vai a Destra"
            # Comandi verticali: richiedono entrambe le mani
            elif all_gestures.count("INDEX_UP") == 2:
                final_command = "Vai Su"
            elif all_gestures.count("THUMB_DOWN") == 2:
                final_command = "Vai Giu"

        # Disegna il comando finale i
        (text_width, text_height), _ = cv2.getTextSize(final_command, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(frame, final_command, ((frame.shape[1] - text_width) // 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Descrive il titolo della finestra
        cv2.imshow('Controllo Drone con Gesti', frame)

        # Chiude il programma (esce dal loop infinito) premendo il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Pulisce le finestre
    cap.release()
    cv2.destroyAllWindows()