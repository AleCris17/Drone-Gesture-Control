import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from pyparrot.Minidrone import Mambo
import time

# -----------------------------
# CONNESSIONE AL DRONE
# -----------------------------
mambo_address = None  # Inserisci MAC se usi BLE, altrimenti Wi-Fi
print("Connessione al Mambo via Wi-Fi...")
mambo = Mambo(mambo_address, use_wifi=True)
success = mambo.connect(num_retries=5)
print(f"Connessione: {success}")

if not success:
    print("Impossibile connettersi al Mambo. Uscita.")
    exit()

mambo.smart_sleep(1)
mambo.ask_for_state_update()
mambo.smart_sleep(1)

# Decollo sicuro
mambo.safe_takeoff(timeout=10)
if mambo.sensors.flying_state not in ["flying", "hovering"]:
    print("Decollo fallito.")
    mambo.disconnect()
    exit()

# Variabile per ritardo atterraggio iniziale
takeoff_start = time.time()

# -----------------------------
# SETUP WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
ret, frame = cap.read()
screen_h, screen_w = frame.shape[:2]

# -----------------------------
# CARICA MODELLO YOLOv8
# -----------------------------
model = YOLO("yolov8n.pt")  # modello locale

# -----------------------------
# BUFFER PER STABILIZZAZIONE COMANDI
# -----------------------------
gesture_history = deque(maxlen=2)  # buffer a 2 frame
confirmed_command = ""

# -----------------------------
# LOOP PRINCIPALE
# -----------------------------
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # effetto specchio

        results = model(frame)
        final_command = ""
        hand_boxes = []

        # -----------------------------
        # ESTRAZIONE DELLE MANI
        # -----------------------------
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                hand_boxes.append(box.cpu().numpy())

        # -----------------------------
        # LOGICA GESTURE 
        # -----------------------------
        if hand_boxes:
            for box in hand_boxes:
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                # Salita: mano alta
                if box_center_y < screen_h / 2:
                    final_command = "UP"
                # Atterraggio: mano molto bassa
                elif box_center_y > 0.8 * screen_h:
                    final_command = "LAND"
                # Movimento laterale
                elif box_center_x < screen_w / 3:
                    final_command = "LEFT"
                elif box_center_x > 2 * screen_w / 3:
                    final_command = "RIGHT"

        # -----------------------------
        # STABILIZZAZIONE COMANDI
        # -----------------------------
        gesture_history.append(final_command)
        if gesture_history.count(final_command) >= 2:
            confirmed_command = final_command
        else:
            confirmed_command = ""

        # -----------------------------
        # IGNORA LAND NEI PRIMI 2 SECONDI DOPO IL DECOLLO
        # -----------------------------
        if confirmed_command == "LAND" and (time.time() - takeoff_start) < 2:
            confirmed_command = ""

        # -----------------------------
        # INVIO COMANDI AL DRONE
        # -----------------------------
        if confirmed_command == "LEFT":
            mambo.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
        elif confirmed_command == "RIGHT":
            mambo.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
        elif confirmed_command == "UP":
            mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=0.5)
        elif confirmed_command == "LAND":
            print("Comando di atterraggio ricevuto.")
            mambo.safe_land(timeout=10)
            break

        # -----------------------------
        # VISUALIZZAZIONE STREAM VIDEO
        # -----------------------------
        display_text = confirmed_command if confirmed_command else "Nessun comando"
        cv2.putText(frame, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("Drone Gesture Control YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("CTRL+C -> Atterraggio di emergenza.")
    mambo.safe_land(timeout=10)

finally:
    cap.release()
    cv2.destroyAllWindows()
    if not mambo.is_landed():
        mambo.safe_land(timeout=10)
    mambo.disconnect()
    print("Chiusura completata.")
