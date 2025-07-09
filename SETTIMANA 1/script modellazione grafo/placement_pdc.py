import networkx as nx
import random

def place_pdcs_greedy(G, max_latency):
    pdcs = set()

    
    pmu_nodes = [n for n in G.nodes if G.degree[n] == 1 and n != "CC"]

    for pmu in pmu_nodes:
        path = nx.shortest_path(G, source=pmu, target="CC", weight="latency") #dijkstra
        total_latency = 0
        #print(f" PMU {pmu} → CC: {path}")

        # Inserisci sempre un PDC nel nodo PMU
        pdcs.add(pmu)
        #print(f" PDC iniziale in {pmu}")

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            latency = G[u][v]["latency"]
            #print(f" {u} → {v}, latenza = {latency}, cumulata = {total_latency + latency}")
            #print(f"  controllo: {total_latency} + {latency} = {total_latency + latency} > {max_latency}?")

            if total_latency + latency > max_latency:
                pdcs.add(u)
                #print(f" PDC intermedio in {u} (superata soglia)")
                total_latency = 0
            else:
                total_latency += latency
    return pdcs

def place_pdcs_random(G, num_pdcs, seed=None): #seed=42 permette di avere gli stessi PDC ogni volta
    if seed is not None:
        random.seed(seed)
    candidate_nodes = [n for n in G.nodes if n != "CC"]

    if num_pdcs > len(candidate_nodes):
        raise ValueError("Il numero di PDC richiesto supera il numero di nodi disponibili.")

    pdcs = set(random.sample(candidate_nodes, num_pdcs))
    return pdcs

def place_pdcs_centrality(G, num_pdcs):
    centrality = nx.closeness_centrality(G, distance="latency")
    centrality.pop("CC", None)

    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

    pdcs = set(sorted_nodes[:num_pdcs])
    return pdcs

def place_pdcs_betweenness(G, num_pdcs):
    centrality = nx.betweenness_centrality(G, weight="latency", normalized=True)
    centrality.pop("CC", None)

    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    return set(sorted_nodes[:num_pdcs])
