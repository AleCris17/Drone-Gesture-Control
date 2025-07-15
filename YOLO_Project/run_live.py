import cv2
from ultralytics import YOLO

# NOTA: Aggiorna questo percorso con quello corretto!
# Controlla nella cartella 'runs/pose/' per trovare l'ultima cartella 'trainX'
MODEL_PATH = 'runs/pose/train/weights/best.pt'

if __name__ == '__main__':
    # Carica il modello personalizzato
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        print(f"Controlla che il percorso '{MODEL_PATH}' sia corretto.")
        exit()

    # Apri la webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read() # legge i frame
        if not ret:
            break

        # Esegui la previsione
        results = model(frame)

        # Disegna lo scheletro della mano sul frame
        annotated_frame = results[0].plot()

        # Mostra il video
        cv2.imshow('Riconoscimento Gesti Mano', annotated_frame)

        # Interrompi con il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()