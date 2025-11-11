import pygame
import numpy as np
import networkx as nx
import random
import colorsys

#adicionar arestas "maleáveis" com plasticidade para que possam de desconectar 

'''
Modelos de plasticidade:

1 - plasticidade Hebbiana contínua
dEij/dt = n(wi*wj - \lambda Eij)


'''


# Inicializa pygame
pygame.init()
width, height = 1200, 900
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Brincar de Deus")
clock = pygame.time.Clock()


#======================= Parâmetros físicos ===========================================
C = 0.1        # constante de difusão
dt = 0.1
noise_amp = 50  # intensidade dos estímulos
num_nodes = 30
#=======================================================================================



# Cria rede aleatória
G = nx.erdos_renyi_graph(num_nodes, 0.2)
pos = nx.spring_layout(G, seed=42)

# Centraliza o grafo na tela
min_x = min(pos[i][0] for i in pos)
max_x = max(pos[i][0] for i in pos)
min_y = min(pos[i][1] for i in pos)
max_y = max(pos[i][1] for i in pos)

for i in pos:
    pos[i] = ((pos[i][0] - min_x) / (max_x - min_x), 
              (pos[i][1] - min_y) / (max_y - min_y))
    pos[i] = (pos[i][0] * 0.7 * width + 0.15 * width, 
              (pos[i][1] * 0.7 * height + 0.15 * height))

A = nx.to_numpy_array(G)
D = np.diag(np.sum(A, axis=1))
L = D - A

# Estados iniciais
w = np.random.rand(num_nodes) * 5

# Para transições suaves de cor
current_colors = [(100, 50, 200) for _ in range(num_nodes)]
target_colors = [(100, 50, 200) for _ in range(num_nodes)]
color_transition_speed = 0.1

# Para hover info
font = pygame.font.SysFont(None, 24)
hovered_node = None

# Piloto automático
auto_pilot = False
auto_pilot_timer = 0
auto_pilot_interval = 30  # frames entre estímulos (0.5 segundos a 60 FPS)

def w_to_color(w_value):
    """Converte valor w para cor usando escala HSV->RGB suave"""
    # Normaliza w para [0, 1] com clipping
    normalized_w = np.clip(w_value / 5.0, 0, 1)
    
    # Usa HSV: Hue varia de azul (0.67) a vermelho (0.0)
    hue = 0.67 - (normalized_w * 0.67)  # Azul -> Vermelho
    saturation = 0.8
    value = 0.8 + (normalized_w * 0.2)  # Leve aumento de brilho
    
    # Converte HSV para RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    return (int(r * 255), int(g * 255), int(b * 255))

def update_colors():
    """Atualiza cores suavemente"""
    global current_colors, target_colors
    
    for i in range(num_nodes):
        current_r, current_g, current_b = current_colors[i]
        target_r, target_g, target_b = target_colors[i]
        
        # Interpolação linear suave
        new_r = current_r + (target_r - current_r) * color_transition_speed
        new_g = current_g + (target_g - current_g) * color_transition_speed
        new_b = current_b + (target_b - current_b) * color_transition_speed
        
        current_colors[i] = (new_r, new_g, new_b)

def get_node_under_mouse(mouse_pos):
    """Retorna o nó sob o cursor do mouse, se houver"""
    for i in G.nodes():
        x, y = pos[i]
        distance = np.sqrt((mouse_pos[0] - x)**2 + (mouse_pos[1] - y)**2)
        if distance <= 12:  # Raio do nó
            return i
    return None

def apply_stimulus():
    """Aplica um estímulo aleatório em um nó"""
    node_idx = random.randint(0, num_nodes-1)
    stimulus = random.uniform(-noise_amp, noise_amp)
    w[node_idx] += stimulus
    return node_idx, stimulus

def draw_graph(w):
    screen.fill((10, 10, 20))
    
    # Atualiza cores alvo baseado nos valores w atuais
    for i in G.nodes():
        target_colors[i] = w_to_color(w[i])
    
    # Atualiza cores atuais suavemente
    update_colors()
    
    # Desenha arestas
    for i, j in G.edges():
        xi, yi = pos[i]
        xj, yj = pos[j]
        pygame.draw.line(screen, (80, 80, 100), (xi, yi), (xj, yj), 1)

    # Desenha nós com cores suaves
    for i in G.nodes():
        x, y = pos[i]
        color = current_colors[i]
        
        # Destaque o nó sob o mouse
        if i == hovered_node:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 16)
            pygame.draw.circle(screen, color, (int(x), int(y)), 14)
        else:
            pygame.draw.circle(screen, color, (int(x), int(y)), 12)
        
        # Desenha borda para nós com w alto
        if w[i] > 3:
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 14, 2)

    # Exibe informações do nó sob o mouse
    if hovered_node is not None:
        node_info = [
            f"Nó {hovered_node}",
            f"w_i = {w[hovered_node]:.2f}",
            f"Grau = {G.degree[hovered_node]}"
        ]
        
        # Fundo para legenda
        x, y = pos[hovered_node]
        pygame.draw.rect(screen, (20, 20, 30), (x + 20, y - 30, 120, 70))
        pygame.draw.rect(screen, (50, 50, 70), (x + 20, y - 30, 120, 70), 1)
        
        # Texto da legenda
        for j, line in enumerate(node_info):
            text = font.render(line, True, (200, 200, 200))
            screen.blit(text, (x + 25, y - 25 + j * 20))

    # Exibe instruções
    instructions = [
        "Pressione ESPAÇO para brincar de Deus",
        "Pressione A para ligar/desligar Piloto Automático",
        "Passe o mouse sobre os nós para ver informações"
    ]
    
    for j, instruction in enumerate(instructions):
        text = font.render(instruction, True, (200, 200, 200))
        screen.blit(text, (20, 20 + j * 25))

    # Legenda de cores
    legend_y = height - 100
    pygame.draw.rect(screen, (30, 30, 40), (20, legend_y - 10, 200, 80))
    pygame.draw.rect(screen, (60, 60, 80), (20, legend_y - 10, 200, 80), 1)
    
    legend_title = font.render("Legenda de Cores:", True, (200, 200, 200))
    screen.blit(legend_title, (30, legend_y))
    
    # Exemplos de cores na legenda
    for i, w_val in enumerate([0, 2.5, 5]):
        color = w_to_color(w_val)
        pygame.draw.circle(screen, color, (40 + i * 60, legend_y + 30), 8)
        w_text = font.render(f"w={w_val}", True, (200, 200, 200))
        screen.blit(w_text, (25 + i * 60, legend_y + 45))
    
    # Indicador do Piloto Automático
    auto_pilot_color = (0, 255, 0) if auto_pilot else (255, 0, 0)
    auto_pilot_text = "PILOTO AUTOMÁTICO: LIGADO" if auto_pilot else "PILOTO AUTOMÁTICO: DESLIGADO"
    pygame.draw.circle(screen, auto_pilot_color, (width - 100, 30), 8)
    auto_text = font.render(auto_pilot_text, True, auto_pilot_color)
    screen.blit(auto_text, (width - 180, 20))

# Loop principal
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    hovered_node = get_node_under_mouse(mouse_pos)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Aplica um estímulo aleatório quando a barra de espaço é pressionada
                node_idx, stimulus = apply_stimulus()
                print(f"Estímulo manual: nó {node_idx} recebeu {stimulus:.2f}")
                
            elif event.key == pygame.K_a:
                # Liga/desliga o piloto automático
                auto_pilot = not auto_pilot
                auto_pilot_timer = 0
                status = "LIGADO" if auto_pilot else "DESLIGADO"
                print(f"Piloto automático {status}")

    # Piloto automático
    if auto_pilot:
        auto_pilot_timer += 1
        if auto_pilot_timer >= auto_pilot_interval:
            node_idx, stimulus = apply_stimulus()
            print(f"Estímulo automático: nó {node_idx} recebeu {stimulus:.2f}")
            auto_pilot_timer = 0




#======================================= O SISTEMA FÍSICO, onde o filho chora e a mãe também ================================================================================


    #adicionar o estímulo direcionado, onde um clique com o mouse sobre um nó gera um estímulo; com a memória por Hopfield puder ser implementada, uma sequência padronizada de estímulos pode ser lembrada pela rede e resultar sempre
    # nos mesmos padrões de difusão entre os nós.
    #a plasticidade minhoca = adicionar a energia da rede de Hopfields
    # E = -\sum{ij}^{arestas} [w_ij * x_i * x_j]
    # x são os valores do parâmetro físico (o estímulo)




 
    # Difusão
    dw = -C * (L @ w)
    w += dw * dt

    # Decaimento natural
    w *= 0.99

    # Renderização
    draw_graph(w)
    pygame.display.flip()
    clock.tick(60)


 #=================================================================================================================================   

pygame.quit()
