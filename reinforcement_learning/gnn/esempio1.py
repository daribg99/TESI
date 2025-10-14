import networkx as nx
from typing import List, Tuple, Dict

class PDCEnv: # Definizione dell'ambiente per la gestione dei PDC: serve per simulare il problema del posizionamento dei PDC su un grafo
    def __init__(self, graph: nx.Graph, pmus: List[int], cc: int):
        self.original_graph = graph
        self.pmus = pmus
        self.cc = cc
        self.nodes = list(graph.nodes)
        self.reset() # Fa ripartire l'ambiente, inizializzando i PDC e lo stato del grafo

    def reset(self): # Resetta l'ambiente, inizializzando i PDC e lo stato del grafo
        self.graph = self.original_graph.copy()
        self.pdcs = set()
        self.covered_pmus = set()
        self.done = False
        return self._get_state()

    def _get_state(self): # Restituisce lo stato attuale dell'ambiente, che include i PDC, PMU e CC ( come dizionario )
        return {
            "pdcs": set(self.pdcs),
            "pmus": set(self.pmus),
            "cc": self.cc
        }

    def _check_validity(self): # Controlla se i vincoli sono rispettati: ogni PMU deve essere coperto da almeno un PDC, i PDC devono essere connessi tra loro, e l'ultimo PDC deve essere connesso al CC
        for pmu in self.pmus:
            if not any(neigh in self.pdcs for neigh in self.graph.neighbors(pmu)):
                return False

        for pdc in self.pdcs:
            neighbors = set(self.graph.neighbors(pdc))
            pdc_neighbors = neighbors.intersection(self.pdcs)
            if pdc not in self._get_first_last_pdcs():
                if len(pdc_neighbors) == 0:
                    return False

        if not any(self.cc in self.graph.neighbors(pdc) for pdc in self.pdcs):
            return False

        return True

    def _get_first_last_pdcs(self): # Restituisce i PDC che sono connessi al primo PMU e all'ultimo CC
        first = {pdc for pdc in self.pdcs if any(n in self.pmus for n in self.graph.neighbors(pdc))}
        last = {pdc for pdc in self.pdcs if self.cc in self.graph.neighbors(pdc)}
        return first.union(last)

    def _compute_latency(self): # Costruisce il sottografo tra PMU e CC, controllando che siano attraversati solo PDC. Infine, calcola la latenza totale
        total_latency = 0
        for pmu in self.pmus:
            try:
                subgraph_nodes = list(self.pdcs.union({pmu, self.cc}))
                subgraph = self.graph.subgraph(subgraph_nodes)
                path = nx.shortest_path(subgraph, source=pmu, target=self.cc, weight='latency')
                print(f"[DEBUG] Path PMU {pmu} → CC: {path}")
                for node in path:
                    if node != pmu and node != self.cc and node not in self.pdcs:
                        print(f"[ERRORE] Nodo non-PDC attraversato: {node}")
                        return float('inf')


                latency = sum(self.graph[u][v]['latency'] for u, v in zip(path[:-1], path[1:]))
                total_latency += latency
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return float('inf')
        return total_latency


    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]: # l'agente sceglie un nodo su cui posizionare un PDC. Se l'azione è valida ( controllo con check_validity precedente ), aggiorna lo stato e calcola la ricompensa
        if action in self.pdcs or self.done or self.graph.nodes[action]["type"] in {"PMU", "CC"}:
            return self._get_state(), -10.0, self.done, {"info": "Invalid action"}

        self.pdcs.add(action)
        self.covered_pmus = {pmu for pmu in self.pmus if any(n in self.pdcs for n in self.graph.neighbors(pmu))}

        all_covered = len(self.covered_pmus) == len(self.pmus)
        valid = self._check_validity()

        latency = self._compute_latency() if valid else float('inf')
        penalty = len(self.pdcs) * 0.5

        reward = -latency - penalty if valid and all_covered else -50.0

        self.done = valid and all_covered and latency < float('inf')

        return self._get_state(), reward, self.done, {}

# === Definizione del grafo ===
G = nx.Graph()

nodes = {
    1: {"type": "PMU"},
    2: {"type": "PMU"},
    3: {"type": "normal"},
    4: {"type": "normal"},
    5: {"type": "normal"},
    6: {"type": "normal"},
    7: {"type": "PMU"},
    8: {"type": "normal"},
    9: {"type": "normal"},
    10: {"type": "normal"},
    11: {"type": "normal"},
    12: {"type": "CC"},
}

for node, attr in nodes.items():
    G.add_node(node, **attr)

edges = [
    (1, 4, 1),
    (2, 4, 3),
    (2, 5, 4),
    (3, 5, 5),
    (3, 6, 1),
    (4, 11, 2),
    (5, 11, 7),
    (5, 6, 2),
    (6, 9, 2),
    (6, 8, 1),
    (6, 3, 1),
    (7, 8, 5),
    (8, 9, 2),
    (9, 10, 1),
    (10, 12, 8),
    (11, 12, 1),
]

for u, v, w in edges:
    G.add_edge(u, v, latency=w)

# === Inizializzazione ambiente ===
pmus = [1, 2, 7]
cc = 12
env = PDCEnv(G, pmus, cc)

# === Esempio di interazione ===
state = env.reset()
print("Stato iniziale:", state)

actions = [4, 11, 8, 9, 10]  # nodi scelti come PDC

for step, node in enumerate(actions):
    state, reward, done, info = env.step(node)
    print(f"\nStep {step + 1}:")
    print(f"Azione: posiziona PDC su nodo {node}")
    print("Stato:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

if done:
    print("\n✅ Obiettivo raggiunto: tutti i PMU connessi al CC rispettando i vincoli.")
else:
    print("\n❌ Episodio incompleto o non valido.")
