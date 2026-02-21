#> 20/02/26x
#> versão raven1.0.0 pré testes
#>tarefas: adicionar funções de geração de gráfico para análise de dados
#> adicionar geração de estímulos
#> adicionar motor gráfico com python


import numpy as np
import matplotlib.pyplot as plt
from networkx.classes import edges
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
            ('circular','circular'): conexoes(lambda_val=0.02,eta=0.01),
            ('triangular','circular'): conexoes(lambda_val=0.03,eta=0.02),
            ('triangular','quadratico'): conexoes(lambda_val=0.04,eta=0.045)

        }

        self.node_types = self._intialize_node_types()

        self.edges = self._initialize_edges()

        self.conexoes = self._initialize_conexoes()

        self.pesos_sinapticos = self._initialize_pesos_sinapticos()

        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.edges)}

        self.history = {
            'fases': [],
            'pesos': [],
            'tempo': []

        }


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

    def calcular_peso_dot(self, i: int, j: int, theta_i: float, theta_j:float) -> float:




        params = self.parametros_conexao(i,j) #chama a função
        edge = tuple(sorted((i,j)))
        w_ij = pesos[edge]
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
            pesos_dict[edge] = state[n_neurons:n_neurons+ idx]

        dotfases = np.zeros(n_neurons)
        for i in range(1, n_neurons+1): #index 1-based
            dotfases[i-1] = self.calcular_theta_dot(i, fases, pesos_dict)

        dotpesos = np.zeros(n_edges)
        for idx, edge in enumerate(self.edges):
            i, j = edge
            dotpesos[idx] = self.calcular_peso_dot(i, j, fases[i-1], fases[j-1])

        return np.concatenate([dotfases, dotpesos])


#====================================== rodar motor gráfico ========================================================


