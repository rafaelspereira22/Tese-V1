############### Problema de Otmização matemática aplicado ao problema caixeiro viajante, estudo de caso: São José dos Campos ##################
########################################### Modulo com as funções essenciais para o método ####################################################

import locale
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
### from prettytable import PrettyTable
from itertools import product
from math import sqrt
import random
import folium
from pyproj import Proj
import itertools
### import gurobipy as gp
### from gurobipy import GRB
# importing googlemaps module
#import googlemaps
import pandas as pd
import geopy.distance as gd
import sys
import time

#################################################### Funções auxiliares ##############################################################

# Automatização do calculo de distância entre coordenadas usando a biblioteca geopy:
def coord_dist(coord_1, coord_2):
    '''
    Função para calcular a distância entre dois pontos usando a geodésica
    
    input: 
        latitude e longitude de dois pontos

    operação:
        lat1 + long1 = coordenada 1
        lat2 + long2 = coordenada 2
        dist geodesica (coordenada 1, coordenada 2)

    output: 
        distância em Km entre as coordenadas 1 e 2
    '''
    
    #coord_1 = (lat1, long1)
    #coord_2 = (lat2, long2)

    return gd.geodesic(coord_1, coord_2).km


# calcula o tempo do percurso que será usando p/ limite do expediente do agente:
def working_day_time(distance, veiculo = 'bike'):
    '''
    Função para calcular o tempo de expediente do agente durante um dia de trabalho

    input: 
        distancia calculada pela função "coord_dist".

    operação: 
        distância/velocidade da bicicleta + tempo de parada em cada público alvo.
    
    output: 
        tempo gasto pelo agente durante o dia.
    '''
    if veiculo == 'bike':
        speed = 10
    if veiculo == 'carro':
        speed = 50
    stop = 1/6
    time = (distance/speed) + stop

    return time
    

# Função para criar o mapa de cada região:
def Region_map(postos_data, publico_data, postos_x, postos_y, publico_x, publico_y, regiao='Norte', cor='blue'):
    '''
    Função para geração do Mapa de SJC por regiões

    input:
        Coord X e Y dos postos e publico alvo, região desejada e cor do publico alvo no mapa

    output: 
        Mapa com os postos (em preto) e público alvo (cor selecionada) da Região selecionada
    '''
    
    # Coordenadas de São José dos Campos
    mapa = folium.Map(
     location = [-23.1880, -45.8670],
     zoom_start = 12
    )

    # Add público alvo ao mapa:
    for i in range (len(publico_y)):
        if publico_data['Região'][i] == regiao:
            folium.CircleMarker(
                location = [publico_y[i], publico_x[i]],
                radius = 0.75,
                fill = True,
                color = cor,
                fill_color="blue",
                fill_opacity = 0.75).add_to(mapa)
            
    # Add postos ao mapa:
    for i in range (len(postos_x)):
        if postos_data['Região'][i] == regiao:
            folium.CircleMarker(
                location = [postos_y[i], postos_x[i]],
                radius = 5,
                fill = True,
                color="black",
                fill_color="black",
                fill_opacity = 1).add_to(mapa)

    return mapa


############################################# Preparação para o banco de dados ########################################################

# Função para usar os pontos e postos de uma região:
def Region_data(postos_data, publico_data, n, regiao='Norte'):
    '''
    Função para gerar uma subregião do banco de dados (Banco de dados na região Central, por exemplo)
    
    input: 
        latitude e longitude, n=índice da coordenada de um posto

    operação:
        mask = escolha de uma subregião (Centro, por exemplo)
        stack das primeiras m coordenadas do público e uma coordenada do posto

    output: 
        coordenadas da subregião com m públicos e posto co índice n
    '''
    
    # Criar uma máscara booleana com base na condição
    mask_posto = postos_data['Região'] == regiao
    mask_publico = publico_data['Região'] == regiao
    #indice = publico_data['Índice'] == regiao
    

    # Aplicar a máscara ao DataFrame
    mask_postos = postos_data[mask_posto]
    mask_publico = publico_data[mask_publico]
    #indice = publico_data[indice]
    l = []
    for k in range(len(mask_publico)+1):
        l.append(k)

    points2 = np.array((mask_publico.Y, mask_publico.X)).T

    points1 = np.array((mask_postos.Y.iloc[n], mask_postos.X.iloc[n])).T

    #points0 = np.array((indice)).T

    # Criando o DataFrame
    stack = np.vstack((points1, points2))
    stack_data = pd.DataFrame(stack, columns=['Latitude', 'Longitude'])
    #stack_data = pd.DataFrame(stack, columns=['Y', 'X'])

    #stack_data_2 = pd.DataFrame(l, columns=['Indices'])
    #result = pd.concat([stack_data_2, stack_data]) 
    stack_data['indices'] = l
    stack_data.index[l]
    
    return stack_data


# Função para usar os pontos e postos de uma região clusterizada:
def Region_data_cluster(postos_data, publico_data, n, regiao='Norte'):
    '''
    Função para gerar uma subregião do banco de dados (Banco de dados na região Central, por exemplo)
    
    input: 
        latitude e longitude, n=índice da coordenada de um posto

    operação:
        mask = escolha de uma subregião (Centro, por exemplo)
        stack das primeiras m coordenadas do público e uma coordenada do posto

    output: 
        coordenadas da subregião com m públicos e posto co índice n
    '''
    
    # Criar uma máscara booleana com base na condição da região
    mask_posto = postos_data['Região'] == regiao
    mask_publico = publico_data['Região'] == regiao
    
    # Aplicar a máscara ao DataFrame para filtrar a região desejada
    mask_postos = postos_data[mask_posto]
    mask_publico = publico_data[mask_publico]
    
    # Adicionar filtro para selecionar todas as linhas cujo cluster seja igual a 'n'
    mask_publico = mask_publico[mask_publico['clusters'] == n]

    # Criar listas de coordenadas
    points2 = np.array((mask_publico.Y, mask_publico.X)).T
    points1 = np.array((mask_postos.Y.iloc[n-1], mask_postos.X.iloc[n-1])).T

    # Criar a pilha de coordenadas
    stack = np.vstack((points1, points2))
    stack_data = pd.DataFrame(stack, columns=['Latitude', 'Longitude'])

    # Adicionar índice das linhas
    stack_data['indices'] = range(len(stack_data))
    
    return stack_data
    
######################################################## Otimizador Held-Karp #################################################################

def calcular_distancias(df, ponto_inicial, raio):
    """
    """
    distancias = []
    coordenadas = list(zip(df['Latitude'], df['Longitude']))
    for i, coord in enumerate(coordenadas):
        distancia = coord_dist(ponto_inicial, coord)
        if distancia <= raio:
            distancias.append((i, distancia))
        
    return distancias
    

def held_karp(dists):
    """
    """
    n = len(dists)
    
    # Inicializar a tabela de custos mínimos
    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)
    
    # Populando a tabela de custos mínimos usando programação dinâmica
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev_bits, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)
    
    # Calculando o custo mínimo para retornar à cidade inicial
    bits = (2**n - 1) - 1
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)
    
    # Reconstruir o caminho
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits
    
    path.append(0)
    return opt, list(reversed(path))
    

def otimizador_held_karp_4(df, raio, max_locais=20, fator_aumento=0.1, veiculo='bike', max_time=5.5):
    """
    Função para otimização utilizando o algoritmo Held-Karp, considerando um limite de tempo total.

    Parâmetros:
    - df: DataFrame contendo as coordenadas das localidades (a primeira linha contém o ponto inicial).
    - raio: Raio inicial de busca.
    - max_locais: Número máximo de locais que o agente pode visitar.
    - fator_aumento: Fator de aumento do raio caso o número de locais dentro do raio seja insuficiente.
    - veiculo: Tipo de veículo utilizado (ex: 'bike' ou 'carro').
    - max_time: Tempo máximo permitido para o caminho.

    Retorna:
    - opt: Otimização realizada pelo algoritmo Held-Karp.
    - caminho_original: Lista de índices representando o caminho otimizado.
    - tempo_total: Tempo total gasto no caminho.
    """
    tempo_total = 0

    # Ponto inicial é a primeira linha do DataFrame
    ponto_inicial = (df.iloc[0]['Latitude'], df.iloc[0]['Longitude'])

    # Loop para ajustar o raio até que tenha no mínimo "max_locais" dentro do raio
    while True:
        # Calcular distâncias para todas as localidades dentro do raio de busca
        locais_no_raio = calcular_distancias(df, ponto_inicial, raio)

        # Se o número de locais no raio for suficiente, sair do loop
        if len(locais_no_raio) >= max_locais:
            break

        # Caso contrário, aumentar o raio por um fator de 0.1
        raio += raio * fator_aumento
        print(f'Novo raio: {raio}')

    # Filtrar os locais que estão dentro do limite de locais que o agente pode visitar
    locais_no_raio = sorted(locais_no_raio, key=lambda x: x[1])[:max_locais]

    # Extrair os índices das localidades selecionadas
    indices = [i for i, _ in locais_no_raio]

    # Montar a matriz de distâncias entre os locais selecionados
    n = len(indices)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = coord_dist(
                (df.iloc[indices[i]]['Latitude'], df.iloc[indices[i]]['Longitude']),
                (df.iloc[indices[j]]['Latitude'], df.iloc[indices[j]]['Longitude'])
            )
            dists[i][j] = dist
            dists[j][i] = dist

    # Resolver o TSP com Held-Karp para os locais selecionados
    opt, caminho = held_karp(dists)

    # Converter o caminho para os índices originais do DataFrame
    caminho_original = [indices[i] for i in caminho]

    # Calcular o tempo total considerando o percurso do agente
    for k in range(1, len(caminho_original)):
        # Calcular a distância entre o ponto atual e o anterior
        dist = coord_dist(
            (df.iloc[caminho_original[k - 1]]['Latitude'], df.iloc[caminho_original[k - 1]]['Longitude']),
            (df.iloc[caminho_original[k]]['Latitude'], df.iloc[caminho_original[k]]['Longitude'])
        )

        # Calcular o tempo gasto para percorrer essa distância
        tempo = working_day_time(dist, veiculo)
        novo_tempo_total = tempo_total + tempo

        # Verificar se o tempo total excede o limite permitido
        if novo_tempo_total > max_time:
            print(f'Tempo máximo excedido: {novo_tempo_total}')
            break

        # Atualizar o tempo total
        tempo_total = novo_tempo_total

    return opt, caminho_original, tempo_total


def filtrar_coordenadas(coordenadas):
    """
    Função para filtrar coordenadas geográficas com uma chance de 20% de não visitar, exceto o primeiro par.

    Parâmetros:
    - coordenadas: lista de tuplas (ou listas) com as coordenadas geográficas (Latitude, Longitude).

    Retorna:
    - no_visit_coord: Lista de coordenadas não visitadas.
    - visit_coord: Lista de coordenadas visitadas.
    """
    no_visit_coord = []
    visit_coord = []

    # Garantir que o primeiro par de coordenadas seja sempre visitado
    visit_coord.append(coordenadas[0])

    # Iterar sobre as coordenadas, exceto a primeira
    for i in range(1, len(coordenadas)):
        # Aplicar probabilidade de 20% de "não visita"
        if random.random() < 0.2:
            no_visit_coord.append(coordenadas[i])
        else:
            visit_coord.append(coordenadas[i])

    return no_visit_coord, visit_coord


# Função para gerar o mapa
def best_way(seq_points):
    '''
    input:
        seq_points = sequência de coordenadas gerada pelo otimizador;

    output:
        mapa = mapa com o melhor caminho gerado pelo otimizador.
    '''
    
    # Mapa de São José dos Campos
    mapa = folium.Map(
     location = [-23.1880, -45.8670],
     zoom_start = 12
    )
    # Coordenadas do ponto de partida
    ponto_partida = seq_points[0]
    
    ### folium.Marker(ponto_partida, popup='Ponto de Partida', icon=folium.Icon(color='green')).add_to(mapa)
    folium.CircleMarker(ponto_partida, radius=5, color='black', fill=True, fill_color='black', fill_opacity=1).add_to(mapa)
    # Adicionar marcadores e linhas para os pontos na sequência de best_path
    for i in range(len(seq_points)-1):
        ponto_atual = seq_points[i]
        proximo_ponto = seq_points[i+1]
        
        # Coordenadas do ponto atual e próximo ponto na sequência ### verificar esse stack_data se é necesário ser uma entrada
        coord_atual = ponto_atual
        prox_coord = proximo_ponto

        folium.PolyLine([coord_atual, prox_coord], color='red').add_to(mapa)
        
        # Adicionar marcador para o ponto atual
        ### folium.Marker(coord_atual, popup=f'Ponto {ponto_atual}', icon=folium.Icon(color='blue')).add_to(mapa)
        # Adicionar marcador para o ponto atual
        if i == 0:
            folium.CircleMarker(coord_atual, 
                                radius=8, 
                                color='black', 
                                fill=True, 
                                fill_color='black', 
                                fill_opacity=1, 
                                popup=f'Ponto {ponto_atual}').add_to(mapa)
        else:
            folium.CircleMarker(coord_atual,
                                radius=5, 
                                color='blue', 
                                fill=True, 
                                fill_color='blue', 
                                fill_opacity=1, 
                                popup=f'Ponto {ponto_atual}').add_to(mapa)
        # Adicionar linha conectando o ponto atual ao próximo ponto
        #folium.PolyLine([coord_atual, prox_coord], color='red').add_to(mapa)

    # Conectar o último ponto ao ponto de partida para fechar o caminho
    coord_ultimo = seq_points[-1]
    coord_primeiro = seq_points[0]
    folium.PolyLine([coord_ultimo, coord_primeiro], color='red').add_to(mapa)

    # Adicionar círculo para o último ponto
    folium.CircleMarker(coord_ultimo, radius=5, color='blue', fill=True, fill_color='blue', fill_opacity=1, popup='Último Ponto').add_to(mapa)

    return mapa

    
################################################################## FIM #######################################################################