import numpy as np
import matplotlib.pyplot as plt

data = np.load("dados_simulacao.npz")

fases = data["fases"]
pesos = data["pesos"]
tempo = data["tempo"]

# norma ao longo do tempo
norma = np.sqrt(
    np.sum(fases**2, axis=1) +
    np.sum(pesos**2, axis=1)
)

plt.plot(tempo, norma)
plt.title("Norma do Estado")
plt.show()