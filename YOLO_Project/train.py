from ultralytics import YOLO

if __name__ == '__main__':
    # Carica il modello base per la stima della posa
    model = YOLO('yolo11n-pose.pt')

    # Avvia l'addestramento usando il file di configurazione ufficiale
    results = model.train(
        data="data.yaml",             # Usa il file ufficiale
        epochs=50,                    # Numero di cicli per un addestramento minimo (100 per uno completo ma troppo dispendioso)
        imgsz=640,                    # Dimensione standard delle immagini
        device=0,                     # Usa la GPU
        #resume=True                  # Continua da dove ha lasciato (../weights/last.pt) ma non funziona
    )

    print("Addestramento completato!")
    print("Il tuo modello personalizzato è in: runs/pose/train/weights/best.pt")