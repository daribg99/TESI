import networkx as nx
import numpy as np
import random

# 1. Crea il grafo
G = nx.Graph()
edges = [
    ("PMU1", 1, 1),
    (1, 2, 2),
    (2, 3, 2),
    (3, "CC", 2),
    ("PMU2", 2, 1)
]

for u, v, latency in edges:
    G.add_edge(u, v, latency=latency)

nodes_pdc = [1, 2, 3]  # azioni possibili
num_states = 2 ** len(nodes_pdc)  # ogni stato è una combinazione di PDC piazzati
num_actions = len(nodes_pdc)

# 2. Parametri Q-learning
Q = np.zeros((num_states, num_actions))
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.2   # esplorazione
episodes = 500  # numero di "partite" di apprendimento
max_latency = 8

# 3. Helper: calcola copertura dei PMU
def is_pmu_covered(pdc_nodes):
    for pmu in ["PMU1", "PMU2"]:
        try:
            path = nx.shortest_path(G, source=pmu, target="CC", weight="latency")
            total_latency = 0
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                total_latency += G[u][v]["latency"]
                if path[i] in pdc_nodes:
                    break
            else:
                return False  # nessun PDC trovato nel path
            if total_latency > max_latency:
                return False
        except:
            return False
    return True

# 4. Stato → indice (es. [1, 0, 1] → 5) -> serve per accedere a Q-table, ad esempio
# stato [1, 0, 1] significa PDC piazzato in nodo 1 e 3
def state_to_index(state):
    return int("".join(map(str, state)), 2)

# 5. Training
for episode in range(episodes):
    state = [0, 0, 0]  # nessun PDC piazzato
    for step in range(5):
        s_idx = state_to_index(state)
        
        # Epsilon-greedy
        if random.random() < epsilon: # con probabilità epsilon, esplora...
            action = random.randint(0, num_actions - 1)
        else: # ... altrimenti sfrutta la politica appresa
            action = np.argmax(Q[s_idx])

        if state[action] == 1:
            reward = -5  # penalità per mossa inutile
        else:
            state[action] = 1
            pdc_nodes = [nodes_pdc[i] for i, v in enumerate(state) if v == 1]
            if is_pmu_covered(pdc_nodes):
                reward = 10
                Q[s_idx, action] += alpha * (reward - Q[s_idx, action])
                break
            else:
                reward = -1
        next_s_idx = state_to_index(state)
        Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])


# 6. Usa la Q-table per decidere
print("\nQ-table finale (righe = stati, colonne = azioni su nodi 1-2-3):")
print("Stato bin |  Nodo 1  |  Nodo 2  |  Nodo 3")
print("-------------------------------------------")
for i in range(num_states):
    binary_state = format(i, f"0{len(nodes_pdc)}b")
    q_values = Q[i]
    print(f"  {binary_state}   |" + "".join([f"  {qv:7.2f} |" for qv in q_values]))

print("\nPolitica appresa (azioni migliori per ogni stato):")
for i in range(num_states):
    binary_state = format(i, f"0{len(nodes_pdc)}b")
    best_action = np.argmax(Q[i])
    print(f" Stato {binary_state} → piazza PDC in nodo {nodes_pdc[best_action]}")