
import pygame
import numpy as np
from dataclasses import dataclass

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

        self.node_types = self._initialize_node_types()
        self.edges = self._initialize_edges()

        self.n_neurons = len(self.node_types)
        self.n_edges = len(self.edges)

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


        #indices vetorizados
        self.edge_i = np.array([e[0] - 1 for e in self.edges])
        self.edge_j = np.array([e[1] - 1 for e in self.edges])

        #vetores \eta e \lambda por aresta
        self.eta_edges = np.zeros(self.n_edges)
        self.lambda_edges = np.zeros(self.n_edges)

        for idx, (i,j) in enumerate(self.edges):
            tipo_i = self.node_types[i]
            tipo_j = self.node_types[j]

            edge_type = tuple(sorted((tipo_i, tipo_j)))
            params = self.parametro_conexao[edge_type]

            self.eta_edges[idx] = params.eta
            self.lambda_edges[idx] = params.lambda_val

        self.omega = np.array([
            self.pesos_naturais[self.node_types[i + 1]]
            for i in range(self.n_neurons)

        ])




        #=============================== estado inicial ===========================================
        theta0 = np.random.uniform(0, 2*np.pi, self.n_neurons)
        w0 = np.random.uniform(0.01, 0.1, self.n_edges)

        self.state = np.concatenate([theta0, w0])
        self.t = 0.0
        self.step_count = 0

        self.history = { #histórico
            'fases': [],
            'pesos': [],
            'tempo': []

        }
    # posições visuais
        self.positions_py = self.kickstart()

#=========================== estrutura da rede ===================================================
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
    def _initialize_edges(self):

        edges = set()

        edges.update([(1,2)])

        edges.update([
            (1,3),(1,4),(1,7),(1,8),
            (2,5),(2,6),(2,9),(2,10)
        ])

        edges.update([
            (7,13),(8,13),(3,11),(4,11),
            (5,12),(6,12),(9,14),(10,14)
        ])

        return sorted([tuple(sorted(e)) for e in edges])




#=========================== SISTEMA DINÂMICO VETORIZADO ==========================================
    def sistema(self, t, state):

        fases = state[:self.n_neurons]
        pesos = state[self.n_neurons:]

        # diferença de fase vetorizada
        delta = fases[self.edge_j] - fases[self.edge_i]

        # ---------------- dinâmica dos pesos ----------------

        dotpesos = (
            self.eta_edges * np.sin(delta)
            - self.lambda_edges * pesos
        )

        # ---------------- dinâmica das fases ----------------

        dotfases = np.zeros(self.n_neurons)

        contrib = pesos * np.sin(delta)

        # soma acumulada eficiente
        np.add.at(dotfases, self.edge_i,  contrib)
        np.add.at(dotfases, self.edge_j, -contrib)

        dotfases += self.omega

        return np.concatenate([dotfases, dotpesos])

# integração euler
    def euler(self,h=0.01):
        derivadas = self.sistema(self.t, self.state)
        self.state += h*derivadas

        self.state[:self.n_neurons] = np.mod(
            self.state[:self.n_neurons],
            2*np.pi
        )

        w_max = 5.0
        self.state[self.n_neurons:] = np.clip(
            self.state[self.n_neurons:],
            -w_max,
            w_max

        )

        self.t += h
        self.step_count += 1

#============================== estatísticas =====================================================

    def norma_estado(self):
        return np.linalg.norm(self.state)

    def matriz_correlacao(self):
        fases = self.state[:self.n_neurons]
        delta = fases[:, None] - fases[None, :]
        return np.cos(delta)

    def estatistica_pesos(self):
        pesos = self.state[self.n_neurons:]
        return {
            "media": np.mean(pesos),
            "variancia": np.var(pesos),
            "max": np.max(pesos),
            'min': np.min(pesos),

        }
    def registrar_estado(self):

        self.history['fases'].append(
            self.state[:self.n_neurons].copy()
        )
        self.history['pesos'].append(
            self.state[self.n_neurons:].copy()
        )
        self.history['tempo'].append(self.t)

# ======================================= instâncias do pygame =============================================================
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
        for idx, edge in enumerate(self.edges):
            i,j = edge
            peso = self.state[self.n_neurons + idx]
            intenisidade = int(
                255*min(abs(peso)/5.0,1.0)

            )
            cor = (intenisidade,50, 255-intenisidade)

            pygame.draw.line(
                tela,cor,self.positions_py[i],
                self.positions_py[j],
                2

            )
        for node, node_type in self.node_types.items():
            x,y = self.positions_py[node]
            if node_type == NeuronType.circular:
                pygame.draw.circle(tela, (120, 170, 255), (x, y), 14)

            elif node_type == NeuronType.triangular:
                points = [(x, y - 12), (x - 12, y + 10), (x + 12, y + 10)]
                pygame.draw.polygon(tela, (120, 255, 170), points)

            elif node_type == NeuronType.quadratico:
                rect = pygame.Rect(x - 12, y - 12, 24, 24)
                pygame.draw.rect(tela, (255, 170, 120), rect)
                
#==================== execução ==========================================================
if __name__ == "__main__":
    pygame.init()
    tela = pygame.display.set_mode((1000, 800))
    clock = pygame.time.Clock()

    net = oscillator_network()
    running = True

    while running:

        tela.fill((15, 15, 25))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        net.euler(0.01)

        if net.step_count % 10 == 0:
            net.registrar_estado()

        net.desenho(tela)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    np.savez(
        "dados_simulacao.npz",
        fases=np.array(net.history['fases']),
        pesos=np.array(net.history['pesos']),
        tempo=np.array(net.history['tempo'])
    )



