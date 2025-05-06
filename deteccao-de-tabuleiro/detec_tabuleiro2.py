"""
Esse programa aplica os conceitos apresentados no detec_tabuleiro.ipynb
e integra eles com a imagem de uma webcam em tempo real. 

q sai do programa
w detecta o tabuleiro
e pergunta por uma casa do tabuleiro
r calcula as casas do tabuleiro
t mostra as linhas 
y mostra as interseções
"""


import cv2 as cv
import numpy as np
import math
from typing import List, Tuple

PENSAR_EM_UM_NOME = False

casas = {}

linhas_ativado = False
pontos_ativado = False

nome_casas = [f"{l}{n}" for l in
               ['a','b','c','d','e','f','g', 'h']
               for n in sorted(range(1, 9), reverse=True)]

# Funções:

def agrupar_linhas_parecidas(linhas: List[List[List[float]]], r_limite: int,
                                    theta_limite: float) -> List[List[Tuple[np.float32]]]:
    """Essa função recebe uma lista de linhas do HoughLines e agrupa
    todas as linhas que estiverem próximas. Essa proximidade é calculada
    através de dois parâmetros: r_limite e theta_limite."""

    grupos = []

    for linha in linhas:
        r_linha, theta_linha = linha[0]
        linha_adicionada = False
        for grupo in grupos:
            for r_grupo, theta_grupo in grupo:
                if abs(r_linha - r_grupo) <= r_limite and abs(theta_linha - theta_grupo) <= theta_limite:
                    grupo.append((r_linha, theta_linha))
                    linha_adicionada = True
                    break
            if linha_adicionada:
                break

        if linha_adicionada == False:
            grupos.append([(r_linha, theta_linha)])
    
    return grupos

def linha_media(lista: List[List[Tuple]]) -> List[Tuple]:
    """Essa função recebes as linhas da função
    agrupar_linhas_parecidas e retorna uma lista
    com a linha média de cada grupo"""
    lista_linhas_medias = []
    for l in lista:
        rs = [i[0] for i in l]
        thetas = [t[1] for t in l]
        r = sum(rs) / len(rs)
        theta = sum(thetas) / len(thetas)
        lista_linhas_medias.append(([r, theta]))

    return lista_linhas_medias

def polarparacartesiana(rho, theta):
    """Essa função recebe os parâmetros de uma equação polar
    e transforma em parâmetros de uma equação cartesiana."""

    a = math.cos(theta)
    b = math.sin(theta)
    c = -rho

    return a, b, c
            
def intersecoes(reta1, reta2):
    """Essa função recebe duas retas e verifica
    se há uma interseção entre elas."""
    
    rho1, theta1 = reta1
    rho2, theta2 = reta2

    a1, b1, c1 = polarparacartesiana(rho1, theta1)
    a2, b2, c2 = polarparacartesiana(rho2, theta2)

    determinante = (a1 * b2) - (b1 * a2)
    if determinante == 0:
        return None
    
    dx = (-c1 * b2) - (-c2 * b1)
    dy = (-c2 * a1) - (-c1 * a2)
    x = int(dx / determinante)
    y = int(dy / determinante)
    
    return x, y

def todosospontos(linhas):
    """Essa função recebe uma lista de linhas e 
    retorna os pontos de  interseções entre elas."""

    pontos = []

    linhas = [line for line in linhas]
    for i in range(len(linhas)):
        for j in range(i + 1, len(linhas)):
            ponto = intersecoes(linhas[i], linhas[j])
            if ponto: pontos.append(ponto)

    return pontos

def contorno_tabuleiro(pts):
    pts = sorted(pts, key= lambda y: y[1])

    pts_11menores = pts[:11]
    pts_11menores = sorted(pts_11menores, key= lambda x: x[0])

    pts_11maiores = pts[-11:]
    pts_11maiores = sorted(pts_11maiores, key= lambda x: x[0])

    # Pontos
    pt1 = pts_11menores[0]
    pt2 = pts_11menores[-1]
    pt3 = pts_11maiores[0]
    pt4 = pts_11maiores[-1]

    cv.line(frame, pt1, pt2, color=(255, 0, 0), thickness=3)
    cv.line(frame, pt2, pt4, color=(255, 0, 0), thickness=3)
    cv.line(frame, pt4, pt3, color=(255, 0, 0), thickness=3)
    cv.line(frame, pt3, pt1, color=(255, 0, 0), thickness=3)

def pontos81(lista: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Essa função recebe os pontos os tabuleiro e remove os pontos que 
    são referentes a borda do tabuleiro, deixando somente os pontos
    que definem as casas do tabuleiro.

    1) Ordenar os pts do menor y para o maior y
    2) Descartar os 11 menores y e os 11 maiores
    3) Pegar os 11 maiores y e ordnar eles pelo x
    4) Pegar os 11 maiores já ordenados e descartar o menor e o maior
    5) Colocar esses 9 pts que sobraram em um dicionário com o menor sendo 1 e o maior sendo 9
    6) Descartar esses pts e fazer tudo de novo a partir do passo 3
    """
    vertices_tabuleiro = []

    lista_y = sorted(lista, key= lambda y: y[1])
    lista_y = lista_y[11:-11]

    for i in range(9):
        _11maiores = lista_y[-11:]
        _11maiores = sorted(_11maiores, key= lambda x: x[0])
        _9maiores = _11maiores[1:-1]
        for i in _9maiores:
            vertices_tabuleiro.append(i)

        lista_y = lista_y[:-11]

    assert len(vertices_tabuleiro) == 81

    return vertices_tabuleiro

def casas_tabuleiro(v: List[Tuple[int, int]]):
    """
    Essa função recebe um dicionário com os 
    vértices do tabuleiro e adiciona o nome da casa
    e suas respectivas coordenadas no dicionário 'casas'. 

    1) Definir cada casa seguindo a seguinte lógica: os pts 1 e 2 + os pts 10 e 11 são
    os limites da casa a1, os pts 3 e 4 + os pts 12 e 13 são os limites da casa b1...
    2) Adicionar as coordendas no dicionário
    """

    coluna_a = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(8)]
    coluna_b = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(9,17)]
    coluna_c = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(18,26)]
    coluna_d = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(27,35)]
    coluna_e = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(36,44)]
    coluna_f = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(45,53)]
    coluna_g = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(54,62)]
    coluna_h = [[v[i], v[i+1], v[i+9], v[i+10]] for i in range(63,71)]

    colunas = []
    colunas.extend(coluna_a)
    colunas.extend(coluna_b)
    colunas.extend(coluna_c)
    colunas.extend(coluna_d)
    colunas.extend(coluna_e)
    colunas.extend(coluna_f)
    colunas.extend(coluna_g)
    colunas.extend(coluna_h)

    for i in range(64):
        casas[nome_casas[i]] = colunas[i]

def identificar_casa(img: np.ndarray, coord: List[Tuple[int, int]]) -> np.ndarray:
    """
    Essa função recebe uma imagem e uma lista com
    quatro coordenadas e muda a cor dos pixels
    dentro dessas quatro coordenadas para 
    vemelho.

    Usaremos a função fillPoly do opencv
    """

    # Correção para a função preencher a casa por completo 
    a,b,c,d = coord
    pontos = [a,b,c,d]
    pontos = np.array(pontos)
    cv.fillPoly(img, pts=[pontos], color=(100, 100, 255))

    pontos = [a,c,b,d]
    pontos = np.array(pontos)
    cv.fillPoly(img, pts=[pontos], color=(100, 100, 255))


# Abrindo webcam
cap = cv.VideoCapture(1)
while True:
    ret, frame = cap.read()

    if not ret:
        print('Não tem frame')
        break

    k = cv.waitKey(1)
    if k == ord('q'):
        break

    if k == ord('w'):
        PENSAR_EM_UM_NOME = True

    if PENSAR_EM_UM_NOME:
        # Transformando a imagem para cinza
        frame_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Aplicando o Canny Edge Detection
        frame_canny = cv.Canny(frame_cinza, 100, 290)

        # Aplicando o Hough Line Transform
        linhas = cv.HoughLines(frame_canny,rho=1,theta=np.pi/180,threshold=120)

        # Calculando interseções

        resultado = agrupar_linhas_parecidas(linhas, 6, np.pi/20) # testar parametros para ver qual é o melhor
        linhas = linha_media(resultado)
        pts = todosospontos(linhas)
        pts = [i for i in pts if (i[0] > 0) and (i[1] > 0) and (i[0] < 2000) and (i[1] < 2000)]


        PENSAR_EM_UM_NOME = False
        

    # Identificando casas do tabuleiro

    if k == ord('r'):
        pontos = pontos81(pts)  
        casas_tabuleiro(pontos)

    if k == ord('e'):
        casa_input = input('digite uma casa do tabuleiro: ')

    # Desenhando linhas nos limites do tabuleiro
    try:
        contorno_tabuleiro(pts)
        identificar_casa(frame, casas[casa_input])
    except Exception as e:
        pass

    # Desenhando linhas nos limites das casas
    if k == ord('t'):
        linhas_ativado = True

    if linhas_ativado:
        for linha in linhas: # forma de desenhar linhas sugerida pela documentação da função
            rho,theta = linha
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(frame,(x1,y1),(x2,y2),(120,0,0),2)

    # Desenhando pontos nas interseções
    if k == ord('y'):
        pontos_ativado = True

    if pontos_ativado:
        for pt in pontos:
            cv.circle(frame, pt, radius=1, color=(120,0,0), thickness=5)
    

    cv.imshow("Detectando tabuleiro", frame)

cap.release()
cv.destroyAllWindows()