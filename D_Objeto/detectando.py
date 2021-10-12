import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # utilizar la wecam

# para utilizar el mdodelo que genera el soft
majinBooClassif = cv2.CascadeClassifier(
    'C:/Users/Usuario/Desktop/proyectos python/IArecognizer/classifier/cascade.xml')

while True:

    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    toy = majinBooClassif.detectMultiScale(img_gray, scaleFactor=9, minNeighbors=91, minSize=(
        50, 50))  # se convierte la imagen en escala de gris y se guarda en variable toy

    for (x, y, w, h) in toy:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),
                      2)  # dibujar el rectangulo
        cv2.putText(frame, 'Benjamin', (x, y-10), 2, 0.7, (0, 255, 0),
                    2, cv2.LINE_AA)  # colocar el nombre del objeto

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
