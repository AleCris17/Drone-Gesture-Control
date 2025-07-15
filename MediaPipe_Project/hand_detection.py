# Importiamo i moduli
import cv2
import mediapipe as mp
import time

# Var per selezionare una webcam
camera = cv2.VideoCapture(0)

# Usiamo la soluzione hand traking da mediapipe
mpHand = mp.solutions.hands         # Riconoscimento delle mani
hands = mpHand.Hands()              # Estraiamo il metodo
mpDraw = mp.solutions.drawing_utils # Strumenti per tracciare lo scheletro della mano

# Var per calcolarci gli fps
previusTime = 0
currentTime = 0

while True:
    # Leggere la videocamera
    success, img = camera.read()

    # convertire i colori da BGR a RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processare l'immagine con la var hands (quindi con il metodo Hands() di mp)
    results = hands.process(imgRGB)

    # Verificare se mp ha riconosciuto delle mani nell'immagine
    if results.multi_hand_landmarks:
        for handLms in  results.multi_hand_landmarks:
            # Disegnare i landmarks sulla mano
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

    # Calcolo fps
    currentTime = time.time()
    fps = 1 / (currentTime - previusTime)
    previusTime = currentTime

    # Mostriamo gli fps sulla finestra
    cv2.putText(img, 'FPS ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,255 ,0), 2)

    # Titolo della finestra
    cv2.imshow("Riconoscimento mani", img)
    cv2.waitKey(1) # Acquisizione frame ogni millisecondo