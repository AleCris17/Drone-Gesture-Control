# Importiamo i moduli
import cv2
import mediapipe as mp
import time

# Var per selezionare una webcam
camera = cv2.VideoCapture(0)

# Usiamo la soluzione hand traking da mediapipe
mpHand = mp.solutions.hands  # Riconoscimento delle mani
# Aumentiamo a max_num_hands=2 per analizzare due mani
hands = mpHand.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils  # Strumenti per tracciare lo scheletro

# Var per calcolarci gli fps
previusTime = 0
currentTime = 0

while True:
    # Leggere la videocamera
    success, img = camera.read()
    img = cv2.flip(img, 1)  # Specchia l'immagine per una visualizzazione naturale

    # Convertire i colori da BGR a RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processare l'immagine
    results = hands.process(imgRGB)

    # Verificare se mp ha riconosciuto delle mani
    if results.multi_hand_landmarks:
        # Itera su ogni mano rilevata
        for idx, handLms in enumerate(results.multi_hand_landmarks):

            # --- INIZIO LOGICA GESTI PER ENTRAMBE LE MANI ---

            myHand = handLms.landmark
            finger_state = []
            tipIds = [4, 8, 12, 16, 20]

            # 1. Ottieni l'etichetta Destra/Sinistra
            handedness_info = results.multi_handedness[idx]
            hand_label = handedness_info.classification[0].label

            # 2. Logica per le 4 dita (uguale per entrambe le mani)
            for id in range(1, 5):
                if myHand[tipIds[id]].y < myHand[tipIds[id] - 2].y:
                    finger_state.append(True)  # Dito Su
                else:
                    finger_state.append(False)  # Dito Giu

            # 3. Logica per il Pollice (diversa per mano destra e sinistra)
            if hand_label == "Right":
                # Il pollice destro e' "su" se la sua punta (4) e' piu a sinistra della sua base (3)
                if myHand[tipIds[0]].x < myHand[tipIds[0] - 1].x:
                    finger_state.insert(0, True)
                else:
                    finger_state.insert(0, False)
            elif hand_label == "Left":
                # Il pollice sinistro e' "su" se la sua punta (4) e' piu a destra della sua base (3)
                if myHand[tipIds[0]].x > myHand[tipIds[0] - 1].x:
                    finger_state.insert(0, True)
                else:
                    finger_state.insert(0, False)

            # 4. Riconoscimento Gesti Specifici
            totalFingers = finger_state.count(True)
            gesture_text = ""

            if totalFingers == 0:
                gesture_text = "Pugno Chiuso"
            elif totalFingers == 1 and finger_state[0]:
                gesture_text = "Pollice Su!"
            elif totalFingers == 2 and finger_state[1] and finger_state[2]:
                gesture_text = "Vittoria!"
            elif totalFingers == 5:
                gesture_text = "Mano Aperta"
            else:
                gesture_text = f"{totalFingers} Dita"

            # Scrivi il gesto riconosciuto vicino al polso
            wrist_pos = (int(myHand[0].x * img.shape[1]), int(myHand[0].y * img.shape[0]))
            cv2.putText(img, gesture_text, (wrist_pos[0], wrist_pos[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255),
                        2)

            # Disegnare i landmarks sulla mano
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

    # Calcolo fps
    currentTime = time.time()
    fps = 1 / (currentTime - previusTime)
    previusTime = currentTime

    # Mostriamo gli fps sulla finestra
    cv2.putText(img, 'FPS ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Titolo della finestra
    cv2.imshow("Riconoscimento mani", img)

    # Exit per la finestra
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()