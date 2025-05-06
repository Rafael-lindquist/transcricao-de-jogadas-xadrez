"""
Esse programa usa o modelo my_model.pt para detectar peças de xadrez
em um frame da webcam

O que esse programa tem que fazer:
    - Abrir a webcam
    - Usar o modelo no frame da webcam
    - Desenhar as boundingboxes com a resposta do modelo

"""

import os

from ultralytics import YOLO
import cv2 as cv

modelo_path = os.path.join("deteccao-de-pecas","my_model","my_model.pt")
modelo = YOLO(modelo_path, task='detect')

labels = modelo.names # dicionário com as labels em ordem alfabética

cores_labels = {0:(164,120,87), 1:(68,148,228), 2:(93,97,209),
                3:(178,182,133), 4:(88,159,106), 5:(96,202,231),
                6:(159,124,168), 7:(169,162,241), 8:(98,118,150),
                9:(172,176,184), 10:(30,30,30), 11:(74,40,200)}

# Abrinco a câmera e exibido o vídeo
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame,(1280, 720))
    if cv.waitKey(1) == ord('q'):
        break

    # Usando o modelo
    resultado = modelo(frame, verbose=False)
    deteccoes = resultado[0].boxes

    # Passando por todas as detecções e pegando as coordenadas, as labels e a confiânça
    contagem_pecas = 0
    for i in range(len(deteccoes)):

        # Coordenadas
        xyxy = deteccoes[i].xyxy.cpu()
        xyxy = xyxy.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Label
        id = int(deteccoes[i].cls.item())
        label = labels[id]

        # Confiança
        confianca = deteccoes[i].conf.item()

        # Desenhando a boundingbox
        if confianca > 0.4: # tenho que testar qual a melhor confiança ainda

            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), cores_labels[id], 2)
            cv.putText(frame, f"{label} {confianca}", (xmin+1, ymin-6), cv.FONT_HERSHEY_COMPLEX_SMALL, .6, (0,0,0), 1)

            contagem_pecas += 1


    cv.putText(frame, f'Numero de pecas: {contagem_pecas}', (10,30), cv.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    cv.imshow("Pensar em um nome", frame)

cap.release()
cv.destroyAllWindows()