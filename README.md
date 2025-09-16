# Drone Gesture Control

## Introduzione
Questo progetto implementa un sistema di controllo di un **Parrot Mambo Minidrone** basato sul riconoscimento dei gesti della mano tramite webcam.  

L’obiettivo principale è confrontare due approcci differenti di visione artificiale:

1. **MediaPipe Hands** – rete neurale ottimizzata di Google per il tracciamento dei landmark della mano.  
2. **YOLO (You Only Look Once)** – modello di object detection in tempo reale.  

Entrambe le implementazioni sono contenute in questa repository, allo scopo di analizzare i vantaggi e gli svantaggi di ciascun approccio e valutarne la convenienza in applicazioni di controllo drone.

---

## Funzionalità
- Riconoscimento delle mani in tempo reale tramite webcam.  
- Traduzione dei gesti in comandi di volo per il drone.  
- Stabilizzazione dei comandi per ridurre i falsi positivi.  
- Supporto a due approcci differenti: **MediaPipe** e **YOLO**.  
- Meccanismi di sicurezza (hovering in assenza di comandi, atterraggio di emergenza).  

---

## Requisiti
- Python 3.12.10 o inferiore
- Webcam  
- Parrot Mambo Minidrone (con realtiva webcam FPV per utilizzare il suo web server)  

### Librerie Python

Le dipendenze sono elencate in `requirements.txt`:

```text
opencv-python
mediapipe
numpy
pyparrot
zeroconf # per connettersi al drone via Wi-Fi
torch # per YOLO
ultralytics # per YOLOv8
```

Installazione:
```bash
pip install -r requirements.txt
```
## Utilizzo
1. Accendere il Parrot Mambo e connettersi alla rete Wi-Fi del drone.
2. Avviare une delle due implementazioni:

### MediaPipe
```bash
python mediapipe_control.py
```

### YOLO
```bash
python YOLO_control.py
```
3. Una finestra mostrerà il feed della webcam con i gesti riconosciuti.
4. I gesti verranno tradotti in comandi di movimento del drone.

## Mappatura dei gesti
| Gesto | Azione drone |
|---|---|
| Indice puntato verso destra | Movimento a destra |
| Indice puntato verso sinistra | Movimento a sinistra |
| Indice puntato verso l'alto | Movimento verso l'alto |
| Pollice verso il basso | Movimento verso il basso |
| Entrambi i pollici in giù | Atterraggio immediato |

## Sicurezza
- I comandi vengono inviati con una durata molto breve (0.2 secondi) per mantenere la stabilità.
- In assenza di un gesto valido, il drone rimane in hovering.
- Il sistema può essere interrotto manualmente premendo il tasto `q` oppure con `CTRL+C` dal terminale.
- È previsto un gesto di atterraggio di emergenza (due pollici in giù).

## Confronto Mediapipe vs YOLO
Il progetto si articola in due implementazioni parallele:

### Mediapipe Hands
Vantaggi:
- Ottimizzato per il tracciamento dei landmark della mano.
- Elevata precisione delle articolazioni.
- Richiede meno risorse computazionali

Svantaggi:
- Limitato agesture basate sulla posizione delle dita.
- Meno flessibile per il riconoscimento di oggetti o contesti complessi.

### YOLO
Vantaggi:
- Rete neurale generalista per l'object detection.
- Possibilità di riconoscere gesture personalizzate tramite dataset dedicato.
- maggiore felssibilità.

Svantaggi:
- Richiede addestramento specifico per le gesture.
- Consumo computazionale superiore.
- Precisione inferiore nei dettagli delle dita rispetto a MediaPipe.

Nota: Nel mio caso specifico, ho utilizzato YOLOv8 nano preaddestrato per riconoscere i bounding box della mano, visto che i landmarks riconosciuti addestrando YOLOv12 pose erano imprecisi, quindi non adatti per il controllo remoto del drone. 



