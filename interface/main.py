import cv2 as cv
import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from detecta_tabuleiro import agrupar_linhas_parecidas, linha_media, polarparacartesiana
from detecta_tabuleiro import intersecoes, todosospontos, contorno_tabuleiro
from detecta_tabuleiro import pontos81, casas_tabuleiro, identificar_casa

from ultralytics import YOLO

modelo_path = os.path.join("deteccao-de-pecas","my_model","my_model.pt")


class Interface:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title("Pensar em um nome")
        janela.geometry("800x530")

        # Configurações
        self.modelo = YOLO(modelo_path, task='detect')
        self.labels = self.modelo.names # Dicionário com as labels em ordem alfabética
        self.cores_labels = {0:(164,120,87), 1:(68,148,228), 2:(93,97,209), # Procurar uma paleta de cores legal
                3:(178,182,133), 4:(88,159,106), 5:(96,202,231),
                6:(159,124,168), 7:(169,162,241), 8:(98,118,150),
                9:(172,176,184), 10:(30,30,30), 11:(74,40,200)} 

        self.casas = {}
        self.nome_casas = [f"{l}{n}" for l in
                    ['a','b','c','d','e','f','g', 'h']
                    for n in sorted(range(1, 9), reverse=True)]
        
        self.casas_pecas_dict = {}
        for i in self.nome_casas:
            self.casas_pecas_dict[i] = ""

        self.jogadas = []

        self.detecta_tabuleiro = False
        self.calcula_casas = False
        self.pontos_ativado = False
        self.linhas_ativado = False
        self.detecta_pecas = False
        self.anotar_jogadas = False
        self.pts = []

        self.cap = cv.VideoCapture(0)
        
        self.label = tk.Label(janela)
        self.label.place(x=5, y=18)
        
        self.atualizar_frame()
        

        # Botão para detectar tabuleiro
        self.btn_detectar_tabueliro = tk.Button(janela, text="Detectar tabuleiro", command=self.ativar_detectar_tabuleiro)
        self.btn_detectar_tabueliro.place(x=650, y=50)

        # Botão para selecionar uma casa
        self.btn_selecionar_casa = tk.Button(janela, text="Selecionar casa", command=self.ativar_selecionar_casa)
        self.btn_selecionar_casa.place(x=650, y=80)

        # Botão para Calcular casas
        self.btn_calcular_casas = tk.Button(janela, text="Calcular casas", command=self.ativar_calcular_casas)
        self.btn_calcular_casas.place(x=650, y=110)

        # Botão para mostrar as linhas
        self.btn_mostrar_linhas = tk.Button(janela, text="Mostrar linhas", command=self.ativar_linhas)
        self.btn_mostrar_linhas.place(x=650, y=140)

        # Botão para mostrar as intersecções
        self.btn_mostrar_intesecoes = tk.Button(janela, text="Mostrar intersecoes", command=self.ativar_interseccoes)
        self.btn_mostrar_intesecoes.place(x=650, y=170)

        # Botão para detectar peças
        self.btn_detectar_pecas = tk.Button(janela, text="Detectar pecas", command=self.ativar_deteccao_pecas)
        self.btn_detectar_pecas.place(x=650, y=200)

        # Botão para começar a anotar as jogadas
        self.btn_anotar_jogadas = tk.Button(janela, text="Anotar jogadas", command= self.ativar_anotar_jogadas)
        self.btn_anotar_jogadas.place(x=650, y=230)

    def atualizar_frame(self):
        ret, self.frame = self.cap.read()
        if ret:
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)

            if self.detecta_tabuleiro:
                frame_cinza = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                frame_canny = cv.Canny(frame_cinza, 100, 290)
                linhas = cv.HoughLines(frame_canny,rho=1,theta=np.pi/180,threshold=120)

                # Calculando interseções e mostrando contorno
                resultado = agrupar_linhas_parecidas(linhas, 8, np.pi/20) # testar parametros para ver qual é o melhor
                linhas = linha_media(resultado)
                pts = todosospontos(linhas) # lista de tuplas
                self.pts = [i for i in pts if (i[0] > 0) and (i[1] > 0) and (i[0] < 2000) and (i[1] < 2000)]
                self.detecta_tabuleiro = False

            if self.pts:
                try:
                    contorno_tabuleiro(self.pts, self.frame)
                except Exception as e:
                    pass

            if self.calcula_casas:
                self.pontos = pontos81(self.pts)  
                casas_tabuleiro(self.pontos, self.casas, self.nome_casas)

            if self.pontos_ativado:
                try:
                    for pt in self.pontos:
                        cv.circle(self.frame, pt, radius=1, color=(120,0,0), thickness=4)
                except Exception as e:
                    print("É necessário calcular as casas primeiro")

            if self.linhas_ativado:
                try:
                    for i, j in zip(range(9), range(-9,-1)):
                        cv.line(self.frame, self.pontos[i], self.pontos[j],(0, 0, 255),1)
                    for i, j in zip(range(0, 73, 9), range(8, 82, 9)):
                        cv.line(self.frame, self.pontos[i], self.pontos[j],(0, 0, 255),1)
                    cv.line(self.frame, self.pontos[8], self.pontos[-1],(0, 0, 255),1)
                except Exception as e:
                    pass

            # Usando o modelo
            resultado = self.modelo(self.frame, verbose=False)
            deteccoes = resultado[0].boxes

            if self.detecta_pecas:
                for i in range(len(deteccoes)):

                    # Coordenadas
                    xyxy = deteccoes[i].xyxy.cpu()
                    xyxy = xyxy.numpy().squeeze()
                    xmin, ymin, xmax, ymax = xyxy.astype(int)

                    # Casa (ponto estimado no centro da base de cada peça)
                    casa = '-'
                    xp = int(((xmax - xmin) / 2) + xmin)
                    yp = int(ymax - ((ymax - ymin) / 4))
                    for j, k in self.casas.items():
                        if xp in range(k[0][0], k[1][0]) and yp in range(k[2][1], k[0][1]):
                            casa = j

                    # Label
                    id = int(deteccoes[i].cls.item())
                    label = self.labels[id]

                    # Confiança
                    confianca = deteccoes[i].conf.item()

                    # Desenhando a boundingbox
                    if confianca > 0.45: # tenho que testar qual a melhor confiança ainda

                        cv.rectangle(self.frame, (xmin, ymin), (xmax, ymax), self.cores_labels[id], 2)
                        cv.putText(self.frame, f"{label} {confianca:.2f} {casa}", (xmin+1, ymin-6), cv.FONT_HERSHEY_COMPLEX_SMALL, .6, (255,255,255), 1)
                        cv.circle(self.frame, (xp, yp), 2, (0,0,255), 4)
                    
                    try:
                        if self.casas_pecas_dict[casa] == label:
                            pass
                        elif self.casas_pecas_dict[casa] != label:
                            self.casas_pecas_dict[casa] = label
                            if self.anotar_jogadas:
                                self.jogadas.append(f"{label[0].upper()}{casa}")
                                print(self.jogadas[-1])
                    except Exception as e:
                        pass

            img = Image.fromarray(self.frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            # Mostrando jogadas na tela
            texto_ultima_jogada = tk.Label(janela, text="Última jogada: ", font=("Arial", 13))
            texto_ultima_jogada.place(x=650, y=320)
            if self.jogadas:
                ultima_jogada = tk.Label(janela, text=self.jogadas[-1], font=("Arial", 13))
                ultima_jogada.place(x=670, y=350)


        self.janela.after(10, self.atualizar_frame)
    
    def sair(self):
        self.cap.release()
        self.janela.destroy()

    def ativar_detectar_tabuleiro(self):
        self.detecta_tabuleiro = True

    def ativar_calcular_casas(self):
        self.calcula_casas = True

    def ativar_interseccoes(self):
        self.pontos_ativado = not self.pontos_ativado

    def ativar_linhas(self):
        self.linhas_ativado = not self.linhas_ativado

    def ativar_selecionar_casa(self):
        casa_input = input('digite uma casa do tabuleiro: ')
        try:
            identificar_casa(self.frame, self.casas[casa_input])
        except Exception as e:
            pass

    def ativar_deteccao_pecas(self):
        self.detecta_pecas = not self.detecta_pecas

    def ativar_anotar_jogadas(self):
        self.anotar_jogadas = not self.anotar_jogadas


janela = tk.Tk()
instacia = Interface(janela)
janela.mainloop()
print(instacia.jogadas)