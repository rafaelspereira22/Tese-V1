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
    
################################################ Otimizador Colônia de Formigas #######################################################

# Função para adicionar imprevisibiidade nas visitas:
def filter_list_with_probability(lst, probability=0.2):
    """
    Filtra os elementos da lista, excluindo cada elemento com uma certa probabilidade. 
    O primeiro elemento da lista não é considerado para exclusão.
    
    Args:
        lst (list): Lista original de elementos.
        probability (float): Probabilidade de excluir cada elemento (0.2 para 20%).
    
    Returns:
        tuple: Duas listas - (lista filtrada com alguns elementos excluídos, lista de elementos excluídos).
    """
    def exclude_with_probability(element, probability):
        return random.choices([element, None], weights=[1 - probability, probability])[0]
    
    if not lst:
        return ([], [])
    
    # Primeiro elemento é mantido sempre
    filtered_list = [lst[0]]
    removed_list = []
    
    for el in lst[1:]:
        result = exclude_with_probability(el, probability)
        if result is None:
            removed_list.append(el)
        else:
            filtered_list.append(result)
    
    return filtered_list, removed_list
    
    
# Colonia de formigas:
def ant_colony_optimization_4(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q, radius, veiculo='bike'):
    '''
    input: 
        points = dados de entrada, onde o ponto 0 é um posto e 1: são so públicos alvo;
        n_ants = número de formigas para a busca;
        n_iterations = números de iterações do algoritmo;
        alpha = 
        beta = 
        evaporation_rate = taxa de evaporação dos feromônios;
        Q = 
        radius = raio de busca para o "next_point" no algoritmo.
        veiculo = qual veículo será usado pelo agente de saúde, alterando a velocidade e tempo do percurso
        
    Operação:
        Alicação do otimizador nos dados de entrada;
        Utilização do melhor caminho para gerar o Mapa.
    output:
        best_path = sequência de índices do melhor caminho 
        mapa = Mapa com o posto (em preto) e público alvo (azul) da Região selecionada e o melhor caminho (em vermelho)
    '''
    
    start_time = time.time()
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        
        paths = []
        path_lengths = []
        distancia_caminho = 0
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = 0  # Comece sempre pelo primeiro ponto
            visited[current_point] = True # visited[current_point]['Índice']
            path = [current_point]
            path_length = 0
            tempo = 0
            tempo_total = 5.5
            
            while False in visited and tempo < tempo_total:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                # Armazenadno as distâncias entre o ponto atual e todos as outras amostras:
                distances = np.array([coord_dist(points[current_point], point) for point in points])

                # Definindo uma vizinhança, escolhendo amostras dentro de um raio:
                neighbors = np.where(distances <= radius)[0]
                    
                for i, unvisited_point in enumerate(unvisited):
                    # Se a amostra estiver dentro do raio:
                    if unvisited_point in neighbors:
                        distancia = coord_dist(points[current_point],points[unvisited_point]) 
                        probabilities[i] = pheromone[current_point, unvisited_point]**alpha / (distancia)**beta
                
                if np.all(probabilities == 0):
                    # Inicialmente, defina o aumento do raio para um valor maior que 1 para garantir um aumento significativo
                    radius_increase_factor = 1.5
                    
                    # Continue aumentando o raio enquanto todas as probabilidades forem zero
                    while np.all(probabilities == 0):
                        # Aumente o raio
                        radius *= radius_increase_factor
                        
                        # Recalcule os vizinhos com o novo raio
                        neighbors = np.where(distances <= radius)[0]
                        ###print('Raio aumentado para:', radius, 'vizinhança', neighbors)
                        
                        # Recalcule as probabilidades com o novo raio
                        for i, unvisited_point in enumerate(unvisited):
                            if unvisited_point in neighbors:
                                distancia = coord_dist(points[current_point], points[unvisited_point])
                                probabilities[i] = pheromone[current_point, unvisited_point]**alpha / (distancia)**beta
                        
                # Normalize as probabilidades
                probabilities /= np.sum(probabilities)

                # Escolha do próximo ponto usando a probabilidade calculada com os ferômonios
                next_point = np.random.choice(unvisited, p=probabilities)

                path.append(next_point)
                path_length += coord_dist(points[current_point], points[next_point]) # Armazenamento de distâncias

                # Condicional para parada levando-se em conta a jornada de trabalho:
                # Se o tempo for menor que o tempo limite:
                if tempo < tempo_total:
                    tempo += working_day_time(path_length, veiculo)
                # caso contrário, soma-se o tempo para ir até o tempo inicial e subtrai-se a distância entre os pontos antes da parada: 
                else:
                    tempo += working_day_time(coord_dist(points[0], points[current_point]), veiculo) - working_day_time(coord_dist(points[best_path[-2]], points[best_path[-1]]), veiculo)
                
                ### print('tempo:', tempo, 'current point', next_point, 'formiga:', ant+1, 'iteration:', iteration+1)

                visited[next_point] = True
                current_point = next_point
    
            paths.append(path)
            path_lengths.append(path_length)
            #tempo_total = tempo + expediente_agente(coord_dist(points[0], points[current_point]))
            ### print('tempo total:',tempo)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

            ###print('best_path', best_path,'iteration:', iteration+1)
        
        pheromone *= evaporation_rate
        
        #distancia_caminho += path_length
        
        for path, path_length in zip(paths, path_lengths):
            if len(path) > 1:  # Verifica se o caminho tem pelo menos dois pontos
                for i in range(len(path) - 1):
                    pheromone[path[i], path[i+1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    
    return best_path[:-1], best_path_length


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