#> 20/02/26x
#> versão raven1.0.0 pré testes
#>tarefas: adicionar funções de geração de gráfico para análise de dados
#> adicionar geração de estímulos
#> adicionar motor gráfico com python

import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class NeuronType:
    circular = "circular"
    triangular = "triangular"
    quadratico = "quadratico"

@dataclass
class conexoes:
    lambda_val: float
    eta:float

class oscillator_network:
    def __init__(self):
        self.pesos_naturais = {
            NeuronType.triangular: 0.4,
            NeuronType.circular: 0.1,
            NeuronType.quadratico: 0.6,

        }

        self.parametro_conexao = {
            tuple(sorted(('circular', 'circular'))): conexoes(lambda_val=0.02, eta=0.01),
            tuple(sorted(('triangular', 'circular'))): conexoes(lambda_val=0.03, eta=0.02),
            tuple(sorted(('triangular', 'quadratico'))): conexoes(lambda_val=0.04, eta=0.045)
        }

        self.node_types = self._initialize_node_types()

        self.edges = self._initialize_edges()


        self.pesos_sinapticos = self._initialize_pesos_sinapticos()

        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.edges)}

        self.history = {
            'fases': [],
            'pesos': [],
            'tempo': []

        }

        self.positions_py = self.kickstart()


    def _initialize_node_types(self) -> Dict[int, str]:

        return {
            1:NeuronType.circular,
            2:NeuronType.circular,
            3:NeuronType.triangular,
            4: NeuronType.triangular,
            5: NeuronType.triangular,
            6: NeuronType.triangular,
            7: NeuronType.triangular,
            8: NeuronType.triangular,
            9: NeuronType.triangular,
            10: NeuronType.triangular,
            11: NeuronType.quadratico,
            12: NeuronType.quadratico,
            13: NeuronType.quadratico,
            14: NeuronType.quadratico,

        }

    # matriz de adjacencia não direcionada
    def _initialize_edges(self) -> List[Tuple[int, int]]:
        edges = set()

        circ_circ = [(1,2)]
        for i,j in circ_circ:
            edge  = tuple(sorted((i,j)))
            edges.add(edge)


        circ_tri =[(1,3),(1,4),(1,7),(1,8),(2,5),(2,6),(2,9),(2,10)]
        for i,j in circ_tri:
            edge  = tuple(sorted((i,j)))
            edges.add(edge)


        tri_quad = [(7,13),(8,13),(3,11),(4,11),(5,12),(6,12),(9,14),(10,14)]
        for i,j in tri_quad:
            edge = tuple(sorted((i, j)))
            edges.add(edge)

        return sorted(list(edges))



    def _initialize_pesos_sinapticos(self) -> Dict[Tuple[int, int], float]: #condicao inicial do sistema
        pesos = {}
        for edge in self.edges:
            pesos[edge] = np.random.uniform(0.01,0.1)
        return pesos

    def get_edge_type(self, i: int, j: int) -> Tuple[str,str]:
        #todas as arestas são não direcionadas

        edge = tuple(sorted((i,j))) #(menor, maior)

        if edge not in self.edge_to_index:
            raise ValueError(f'Aresta entre {i} e {j} nao existe')

        #tipo de neuronio
        type_i = self.node_types[i]
        type_j = self.node_types[j]

        if type_i == type_j:
            return type_i, type_j
        else:
            return tuple(sorted((type_i, type_j)))


    def parametros_conexao(self, i: int, j: int ) -> conexoes:
        edge_type = self.get_edge_type(i,j)
        return self.parametro_conexao[edge_type]
#======================================================================================================================
#================================ SISTEMA FÍSICO REGENTE ==============================================================
# > baseado diretamente nos documentos em drive.google.com/drive/u/_codas/biblioteca/estudo/_autoria/Raven/DPT
# > a saber: RADPTC1001, RADPTC1002, RADPTC1003 E RADPTC1004.
# > \dot\theta_i = w_i + \sum{j}^{N} w_{ij} * sen(\delta \theta) (1) --> regime de oscilação de fase
# > \dot\ w_ij = \eta * \delta \theta - \lambda * w_ij (2) --> função peso, elemento da matriz W = {w_ij}
# > Z = \eta / \lambda
# > w_ij(*) = Z * sen(\delta \theta) -> \therefore ponto fixo para análise posterior
#

    def calcular_peso_dot(self, i, j, theta_i, theta_j, pesos_dict):

        params = self.parametros_conexao(i,j) #chama a função
        edge = tuple(sorted((i,j)))
        w_ij = pesos_dict[edge]
        Z = params.eta / params.lambda_val #calcula o Z = η/λ para análise dos pontos de equilíbrio
        delta_theta = theta_j - theta_i #calcula a diferença de fase entre nós conectados
        w_ij_dot = params.eta * np.sin(delta_theta) - params.lambda_val * w_ij
        return w_ij_dot

    def calcular_theta_dot(self, i: int , fases: np.ndarray, pesos: Dict ) -> float: #calcular derivada da função fase

        w_i = self.pesos_naturais[self.node_types[i]] #cata o tipo de neurônio
        soma_acoplamento = 0.0 #

        for edge in self.edges:
            if i in edge:
                j = edge[0] if edge[1] == i else edge[1] #encontra o outro nó da aresta

                w_ij = pesos[edge]

                delta_theta = fases[j-1] - fases[i-1] #indice 0-based

                soma_acoplamento += w_ij * np.sin(delta_theta)
        return w_i + soma_acoplamento

    def sistema(self, t:float, state: np.ndarray) -> np.ndarray:
        n_neurons = len(self.node_types)
        n_edges = len(self.edges)

        fases = state[:n_neurons]

        pesos_dict = {}

        for idx, edge in enumerate(self.edges):
            pesos_dict[edge] = state[n_neurons+ idx]

        dotfases = np.zeros(n_neurons)
        for i in range(1, n_neurons+1): #index 1-based
            dotfases[i-1] = self.calcular_theta_dot(i, fases, pesos_dict)

        dotpesos = np.zeros(n_edges)
        for idx, edge in enumerate(self.edges):
            i, j = edge
            dotpesos[idx] = self.calcular_peso_dot(i, j, fases[i-1], fases[j-1],pesos_dict)

        return np.concatenate([dotfases, dotpesos])

#=========================================== integração para análise de dados com range-kutta ===================================================

#    def simular(self, t_final=200, n_points=5000):
#        n_neurons = len(self.node_types)
#        n_edges = len(self.edges)
#        theta_0 = np.random.uniform(0,2*np.pi,n_neurons)
#        w0 = np.array([self.pesos_sinapticos[edge] for edge in self.edges])
#        estado_inicial = np.concatenate([theta_0, w0])
#        t_eval = np.linspace(0, t_final, n_points)
#        solucao = solve_ivp(self.sistema, (0,t_final), estado_inicial, t_eval=t_eval,method="RK-45")
#        return solucao

#====================================== integração para pygame loop com euler pequeno ========================================================
    def kickstart(self, width=1000, height=800):

        x_centro = width // 2
        y_centro = height // 2

        positions = {}

        # Circulares
        offset = 40
        positions[1] = (x_centro - offset, y_centro)
        positions[2] = (x_centro + offset, y_centro)

        # Triangulares
        raio_centro = 180
        for idx, node in enumerate(range(3, 11)):
            angulo = 2 * np.pi * idx / 8
            x_pos = x_centro + raio_centro * np.cos(angulo)
            y_pos = y_centro + raio_centro * np.sin(angulo)
            positions[node] = (int(x_pos), int(y_pos))

        # Quadrangulares
        raio_exterior = 300
        for idx, node in enumerate(range(11, 15)):
            angulo = 2 * np.pi * idx / 4
            x_pos = x_centro + raio_exterior * np.cos(angulo)
            y_pos = y_centro + raio_exterior * np.sin(angulo)
            positions[node] = (int(x_pos), int(y_pos))

        return positions


    def desenho(self,tela):
        for edge in self.edges:
            i, j = edge
            pygame.draw.line(
                tela,
                (100,100,100),
                self.positions_py[i],
                self.positions_py[j],
                2
            )

        for node, node_type in self.node_types.items():
            x,y = self.positions_py[node]

            if node_type ==  NeuronType.circular:
                pygame.draw.circle(tela,(120,170,255),(x,y),14)
            elif node_type ==  NeuronType.triangular:
                points = [
                    (x,y-12),
                    (x-12, y+10),
                    (x+12, y+10)
                ]
                pygame.draw.polygon(tela,(120,255,170),points)
            elif node_type == NeuronType.quadratico:
                rect = pygame.Rect(x-12,y-12,24,24)
                pygame.draw.rect(tela,(255,170,120),rect)


##teste isolado
pygame.init()
tela = pygame.display.set_mode((1000,800))
clock = pygame.time.Clock()

net = oscillator_network()
running = True
while running:
    tela.fill((15,15,25))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    net.desenho(tela)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
