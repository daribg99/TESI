import networkx as nx
import numpy as np
import random

# 1. Crea il grafo esteso
G = nx.Graph()
edges = [
    ("PMU1", 1, 1),
    (1, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (4, 5, 2),
    (5, "CC", 2),
    ("PMU2", 2, 1),
    ("PMU3", 3, 2),
    ("PMU4", 4, 3)
]

for u, v, latency in edges:
    G.add_edge(u, v, latency=latency)

# 2. Nodi candidati per PDC
nodes_pdc = [1, 2, 3, 4, 5]
num_states = 2 ** len(nodes_pdc)
num_actions = len(nodes_pdc)

# 3. Parametri Q-learning
Q = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000
max_latency = 8

# 4. Verifica copertura PMU
def is_pmu_covered(pdc_nodes):
    for pmu in ["PMU1", "PMU2", "PMU3", "PMU4"]:
        try:
            path = nx.shortest_path(G, source=pmu, target="CC", weight="latency")
            total_latency = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                total_latency += G[u][v]["latency"]
                if path[i] in pdc_nodes:
                    break
            else:
                return False
            if total_latency > max_latency:
                return False
        except:
            return False
    return True

# 5. Stato → indice
def state_to_index(state):
    return int("".join(map(str, state)), 2)

# 6. Training con penalità per troppi PDC
for episode in range(episodes):
    state = [0] * len(nodes_pdc)
    for step in range(10):
        s_idx = state_to_index(state)
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(Q[s_idx])

        if state[action] == 1:
            reward = -5
        else:
            state[action] = 1
            pdc_nodes = [nodes_pdc[i] for i, v in enumerate(state) if v == 1]
            if is_pmu_covered(pdc_nodes):
                reward = 10 - 2 * (len(pdc_nodes) - 1)  # penalità per ogni PDC in più
                Q[s_idx, action] += alpha * (reward - Q[s_idx, action])
                break
            else:
                reward = -1
        next_s_idx = state_to_index(state)
        Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])

# 7. Stampa Q-table
print("\nQ-table finale (righe = stati, colonne = azioni su nodi 1-2-3-4-5):")
print("Stato bin | " + " | ".join([f"Nodo {n}" for n in nodes_pdc]))
print("-" * 60)
for i in range(num_states):
    binary_state = format(i, f"0{len(nodes_pdc)}b")
    q_values = Q[i]
    q_str = " | ".join([f"{qv:7.2f}" for qv in q_values])
    print(f"{binary_state} | {q_str}")

# 8. Politica appresa
print("\nPolitica appresa (azione migliore per ogni stato):")
for i in range(num_states):
    binary_state = format(i, f"0{len(nodes_pdc)}b")
    best_action = np.argmax(Q[i])
    print(f" Stato {binary_state} → piazza PDC in nodo {nodes_pdc[best_action]}")
