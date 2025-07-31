import networkx as nx
import random
import numpy as np
from itertools import islice
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
def draw_graph(G, pdcs=None, paths=None, max_latency=None):
    if pdcs is None:
        pdcs = set()

    plt.figure(figsize=(14, 10))

    try:
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
    except Exception:
        print("⚠️ Errore con pydot. Uso spring_layout.")
        pos = nx.spring_layout(G, seed=42)

    edge_labels = nx.get_edge_attributes(G, "latency")
    node_colors = []
    node_labels = {}
    node_edgecolors = []

    for n in G.nodes:
        role = G.nodes[n].get("role")
        label = n

        if n in pdcs:
            color = "orange"
            label += f"\n{G.nodes[n].get('processing', 0)}"
            edge_color = "black"
        elif role == "CC":
            color = "red"
            label += "\n(CC)"
            edge_color = "black"
        elif role == "PMU":
            color = "lightgreen"
            label += "\n(PMU)"
            edge_color = "black"
        else:
            color = "lightblue"
            edge_color = "gray"

        node_colors.append(color)
        node_labels[n] = label
        node_edgecolors.append(edge_color)

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           edgecolors=node_edgecolors,
                           node_size=1100,
                           linewidths=1.8)

    # Disegna tutti gli archi base in grigio
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color="lightgray")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    # Colori diversi per ogni path PMU → CC
    if paths:
        colors = [
            "crimson", "darkgreen", "royalblue", "goldenrod",
            "purple", "darkorange", "deeppink", "teal", "brown"
        ]
        color_map = {}

        for i, (pmu, data) in enumerate(paths.items()):
            path = data["path"]
            delay = data["delay"]
            color = colors[i % len(colors)]
            color_map[pmu] = color
            edges = list(zip(path, path[1:]))

            nx.draw_networkx_edges(G, pos,
                                   edgelist=edges,
                                   width=2.8,
                                   edge_color=color)

        # Testo con le latenze
        text = "Latenze PMU → CC:\n"
        text += "Max latency: " + str(max_latency) + " ms\n"
        for pmu, data in paths.items():
            delay = data["delay"]            
            text += f"{pmu} → CC: {delay:.2f} ms"
            if max_latency is not None and delay > max_latency:
                text += f" ⚠️\n"
            else:
                text += " ✔️\n"

        plt.gcf().text(0.05, 0.85, text, fontsize=9, verticalalignment='top',
                       bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Legenda
    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="CC"),
        Patch(facecolor="lightgreen", edgecolor="black", label="PMU"),
        Patch(facecolor="orange", edgecolor="black", label="PDC (selezionati)"),
        Patch(facecolor="lightblue", edgecolor="gray", label="Altro nodo (candidato)")
    ]
    plt.legend(handles=legend_elements, loc="lower left", fontsize=9, frameon=True)

    plt.title("Grafo con ruoli e path evidenziati", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    
def create_graph(seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()

    # Nodo centrale
    G.add_node("CC", group="core", level=0, processing=10, memory=32, storage=1000,
               status="online", energy=1.0, role="CC")

    # Nodi candidati (potranno diventare PDC o restare inutilizzati)
    for i in range(1, 13):
        G.add_node(f"N{i}",
                   group=random.choice(["A", "B"]),
                   level=random.choice([1, 2]),
                   processing=random.randint(3, 6),
                   memory=random.choice([8, 16]),
                   storage=random.choice([250, 500]),
                   status="online",
                   energy=round(random.uniform(0.85, 0.95), 2),
                   role="candidate")

    # PMU (nodi foglia di livello 3)
    for i in range(1, 4):
        G.add_node(f"PMU{i}",
                   group="A",
                   level=3,
                   processing=1,
                   memory=2,
                   storage=64,
                   status="online",
                   energy=round(0.7 + i * 0.01, 2),
                   role="PMU")

    # Definizione archi
    edges = set([
        ("CC", "N1"), ("CC", "N2"), ("CC", "N3"),
        ("N1", "N2"), ("N2", "N3"), ("N3", "N4"),
        ("N4", "N5"), ("N5", "N6"), ("N6", "N1"),
        ("N1", "N7"), ("N2", "N8"), ("N3", "N9"),
        ("N4", "N10"), ("N5", "N11"), ("N6", "N12"),
        ("N7", "N8"), ("N8", "N9"), ("N9", "N10"),
        ("N10", "N11"), ("N11", "N12"), ("N12", "N7"),
        ("N7", "PMU1"), ("N8", "PMU2"), ("N9", "PMU3"),
        ("N5", "PMU1"), ("N4", "PMU2"), ("N3", "PMU3"),
        ("N2", "N11"), ("N6", "N9")
    ])

    for u, v in edges:
        latency = round(random.uniform(2, 9), 2)
        bandwidth = random.choice([100, 200, 500, 1000])
        status = "up"
        link_type = random.choices(["fiber", "ethernet", "wireless"], weights=[0.4, 0.4, 0.2])[0]

        G.add_edge(u, v,
                   latency=latency,
                   bandwidth=bandwidth,
                   status=status,
                   type=link_type)
    return G

def modify_latency(G):
    while True:
            print("\n🔗 Latenze attuali:")
            for u, v, data in G.edges(data=True):
                print(f"{u} – {v}: {data['latency']} ms")

            risposta = input("\nVuoi modificare una latenza? (s/n): ").lower()
            if risposta != "s":
                break

            u = input("Nodo 1 dell’arco: ").strip()
            v = input("Nodo 2 dell’arco: ").strip()

            if G.has_edge(u, v):
                try:
                    nuova_latenza = float(input(f"Inserisci nuova latenza per l’arco {u}–{v}: "))
                    G[u][v]["latency"] = nuova_latenza
                    print(f"✔️ Latenza aggiornata per {u}–{v} a {nuova_latenza} ms.")
                except ValueError:
                    print("❌ Valore non valido.")
            else:
                print("❌ L’arco specificato non esiste.")
                
def modify_edge_status(G):
    while True:
        print("\n🔗 Stato attuale degli archi:")
        for u, v, data in G.edges(data=True):
            print(f"{u} – {v}: {data['status']}")

        risposta = input("\nVuoi modificare lo stato di un arco? (s/n): ").lower()
        if risposta != "s":
            break

        u = input("Nodo 1 dell’arco: ").strip()
        v = input("Nodo 2 dell’arco: ").strip()

        if G.has_edge(u, v):
            nuovo_stato = input(f"Inserisci nuovo stato per l’arco {u}–{v} (up/down): ").strip().lower()
            if nuovo_stato in ["up", "down"]:
                G[u][v]["status"] = nuovo_stato
                print(f"✔️ Stato aggiornato per {u}–{v} a {nuovo_stato}.")
            else:
                print("❌ Stato non valido. Usa 'up' o 'down'.")
        else:
            print("❌ L’arco specificato non esiste.")

def place_pdcs_greedy(G, max_latency):
    pdcs = set()
    pmu_paths = {}

    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    pmu_to_path = {}

    for pmu in pmu_nodes:
        try:
            paths = nx.shortest_simple_paths(G, source=pmu, target="CC", weight="latency")
            for path in paths:
                # Check validità: tutti i nodi online e archi up
                if all(G.nodes[n].get("status") == "online" for n in path) and \
                   all(G[u][v].get("status") == "up" for u, v in zip(path, path[1:])):
                    pmu_to_path[pmu] = path
                    break
            else:
                print(f"Nessun path valido trovato da {pmu} a CC.")
        except nx.NetworkXNoPath:
            print(f"Nessun path esistente da {pmu} a CC.")
            continue

    for path in pmu_to_path.values(): # trova i PDC
        for node in path[1:-1]:  # Escludi PMU e CC
            if G.nodes[node].get("role") not in {"PMU", "CC"}:
                pdcs.add(node)

    # Calcola i ritardi
    for pmu, path in pmu_to_path.items():
        total_delay = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += G[u][v]["latency"]

        for node in path:
            if node in pdcs:
                total_delay += G.nodes[node].get("processing", 0)

        pmu_paths[pmu] = {"path": path, "delay": total_delay}
    if not pmu_paths:
        print("⚠️ Nessun path valido per alcun PMU.")
        return pdcs

    max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
    max_delay = max_pmu[1]["delay"]
    max_path = max_pmu[1]["path"]

    print("\n Path PMU → CC e relative latenze:")
    for pmu, data in pmu_paths.items():
        print(f"  {pmu} → CC: {data['path']} | Ritardo totale = {data['delay']:.2f} ms")

    print()
    if max_delay > max_latency:
        print(f" Ritardo massimo {max_delay:.2f} ms supera la soglia {max_latency} ms.")
        print(f" Causato dal path: {max_pmu[0]} → CC = {max_path}")
    else:
        print(f" Ritardo massimo {max_delay:.2f} ms sotto la soglia {max_latency} ms.")

    return (pdcs, pmu_paths, max_latency)


def place_pdcs_random(G, max_latency, seed=None):
    if seed is not None:
        random.seed(seed)

    pdcs = set()
    pmu_paths = {}
    pmu_nodes = [n for n in G.nodes if G.nodes[n].get("role") == "PMU"]

    def dfs_random_path(current, target, visited):
        if current == target:
            return [current]

        visited.add(current)
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in visited and G.nodes[neighbor].get("role") != "PMU":
                if G.nodes[neighbor].get("status") != "online":
                    continue
                if G[current][neighbor].get("status") != "up":
                    continue

                path = dfs_random_path(neighbor, target, visited)
                if path:
                    return [current] + path

        visited.remove(current)
        return None

    for pmu in pmu_nodes:
        path = dfs_random_path(pmu, "CC", set())
        if path is None:
            print(f" Nessun cammino valido trovato da {pmu} a CC (senza PMU/offline/down).")
            continue
        total_delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_delay += G[u][v]["latency"]
            if G.nodes[u].get("role") == "candidate":
                total_delay += G.nodes[u].get("processing", 0)

        pmu_paths[pmu] = {"path": path, "delay": total_delay}
        for node in path[1:-1]:
            pdcs.add(node)

    # Trova il path con ritardo massimo
    if pmu_paths:
        max_pmu = max(pmu_paths.items(), key=lambda x: x[1]["delay"])
        max_delay = max_pmu[1]["delay"]
        max_path = max_pmu[1]["path"]

        print("\n Cammini PMU → CC e ritardi:")
        for pmu, data in pmu_paths.items():
            print(f"  {pmu} → CC: {data['path']} | Ritardo = {data['delay']:.2f} ms")

        print()
        if max_delay > max_latency:
            print(f" Ritardo massimo {max_delay:.2f} ms supera la soglia {max_latency} ms.")
            print(f"   Causato dal path: {max_pmu[0]} → CC = {max_path}")
        else:
            print(f" Ritardo massimo {max_delay:.2f} ms sotto la soglia {max_latency} ms.")
    else:
        print(" Nessun path valido trovato da alcun PMU al CC.")

    return (pdcs, pmu_paths, max_latency)

def place_pdcs_bruteforce(G, max_latency):
    
    def is_valid_chain(path, pdc_nodes, G):
        if not path:
            return False

        node_role = [G.nodes[n].get("role") for n in path]
        if node_role[0] != "PMU" or node_role[-1] != "CC":
            return False

        # Tutti i nodi intermedi devono essere PDC validi
        for i in range(1, len(path) - 1):
            if path[i] not in pdc_nodes:
                return False
            if not G.nodes[path[i]].get("online", True):
                return False

        # Verifica adiacenza PDC consecutivi
        for u, v in zip(path, path[1:]):
            if not G.has_edge(u, v) or G[u][v].get("status", "up") != "up":
                return False

        return True

    def compute_path_latency(path, G, pdc_nodes):
        """Calcola la latenza totale del path: somma archi + processing nodi PDC"""
        latency = 0
        for u, v in zip(path, path[1:]):
            latency += G[u][v].get("latency", 0)

        for n in path:
            if n in pdc_nodes:
                latency += G.nodes[n].get("processing", 0)

        return latency

    pmu_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "PMU"]
    candidate_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "candidate"]
    cc_node = [n for n, d in G.nodes(data=True) if d.get("role") == "CC"][0]

    best_config = None
    best_total_latency = float('inf')
    best_paths={}

    for k in range(1, len(candidate_nodes) + 1):
        for pdc_nodes in combinations(candidate_nodes, k): # combinations ritorna tutte le combinazioni di k PDC
            all_covered = True
            total_latency = 0
            current_path={}
            for pmu in pmu_nodes:
                
                allowed_nodes = set(pdc_nodes) | {pmu, cc_node}
                subgraph = nx.Graph()
                for u in allowed_nodes:
                    subgraph.add_node(u, **G.nodes[u])

                for u, v in G.edges():
                    if u in allowed_nodes and v in allowed_nodes:
                        if G[u][v].get("status", "up") == "up":
                            subgraph.add_edge(u, v, **G[u][v])


                try:
                    path = nx.shortest_path(subgraph, source=pmu, target=cc_node, weight="latency")
                except nx.NetworkXNoPath:
                    all_covered = False
                    break

                if not is_valid_chain(path, pdc_nodes, G):
                    all_covered = False
                    break

                latency = compute_path_latency(path, G, pdc_nodes)
                total_latency += latency
                current_path[pmu] = {"path": path, "delay": latency}


            if all_covered and total_latency < best_total_latency:
                best_config = list(pdc_nodes)
                best_total_latency = total_latency
                best_paths = current_path
    
    print("\n📍 Migliori path PMU → CC con configurazione PDC ottima:")
    for pmu, data in best_paths.items():
        path = data["path"]
        delay = data["delay"]
        print(f"{pmu} → CC: {' → '.join(path)}, Ritardo = {delay:.2f} ms")

    return best_config, best_paths, max_latency if best_config else []

def q_learning_placement(G, max_latency, episodes=15000, alpha=0.1, gamma=0.9, epsilon=0.8, seed=None, verbose=False):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes_pdc = [n for n, data in G.nodes(data=True)
                 if data.get("role") == "candidate" and data.get("status") == "online"]
    pmu_nodes = [n for n, data in G.nodes(data=True) if data.get("role") == "PMU"]

    num_actions = len(nodes_pdc)
    Q = np.zeros((2 ** num_actions, num_actions)) # inizializzazione Q-table

    def compute_total_delay(path): # calcola il ritardo totale di un path
        delay = 0.0
        for u, v in zip(path, path[1:]):
            delay += G[u][v]["latency"]
        for node in path[1:-1]:
            delay += G.nodes[node].get("processing", 0)
        return delay
    
    # def valid_path(path, state):# controlla che ci siano solo PDC online e archi up
    #     for node in path[1:-1]:
    #         if node not in nodes_pdc:
    #             return False
    #         idx = nodes_pdc.index(node)
    #         if state[idx] != 1:  # se nodo è candidato ma non attivo ( da state[idx] )
    #             return False
    #     for u, v in zip(path, path[1:]): # se qualche arco nel path non è attivo
    #         if G[u][v].get("status") != "up":
    #             return False
    #     if any(G.nodes[n].get("status") != "online" for n in path):
    #         return False
    #     return True
    
    def valid_path(path, state):
        # Verifica che i nodi intermedi siano PDC attivi
        for node in path[1:-1]:
            if node not in nodes_pdc:
                return False
            idx = nodes_pdc.index(node)
            if state[idx] != 1:
                return False
        # Verifica che tutti gli archi siano up
        for u, v in zip(path, path[1:]):
            if G[u][v].get("status") != "up":
                return False
        # Verifica che tutti i nodi siano online
        if any(G.nodes[n].get("status") != "online" for n in path):
            return False
        return True

    
    def find_best_paths(state):
        pmu_to_best = {} # dizionario nella forma {PMU: (path, delay)}
        for pmu in pmu_nodes:
            best_path = None
            best_delay = float("inf")
            try:
                paths = islice(nx.all_simple_paths(G, pmu, "CC", cutoff=10), 150) # trova fino a 100 cammini, cutoff a 10 nodi per evitare cicli infiniti
                for path in paths:
                    if not valid_path(path, state):
                        continue
                    delay = compute_total_delay(path)
                    if delay < best_delay:
                        best_delay = delay
                        best_path = path
            except nx.NetworkXNoPath:
                continue
            if best_path:
                pmu_to_best[pmu] = {"path": best_path, "delay": best_delay}
        return pmu_to_best

    def state_to_index(state):
        return int("".join(str(b) for b in state), 2)

    best_state = None
    best_score = -float("inf")

    for ep in range(episodes):
        if ep % 1000 == 0: print(f"🔄 Episodio {ep}/{episodes}...")
        state = [0] * num_actions
        pmu_covered = set()

        for step in range(num_actions): # ad ogni step, si attiva un PDC
            s_idx = state_to_index(state) # converte lo stato in indice per Q-table
            available = [i for i in range(num_actions) if state[i] == 0] # seleziona i PDC ancora spenti
            if not available:
                break

            if random.random() < epsilon:
                action = random.choice(available)
            else:
                q_values = Q[s_idx]
                q_masked = [q if i in available else -np.inf for i, q in enumerate(q_values)]
                action = int(np.argmax(q_masked))

            state[action] = 1

            best_paths = find_best_paths(state)
            new_covered = set(best_paths.keys()) # nuove PMU coperte
            delta_covered = len(new_covered - pmu_covered)
            pmu_covered = new_covered

            total_delay = sum(data["delay"] for data in best_paths.values())

            pdc_count = sum(state)

            reward = (
                +20 * delta_covered
                - 1.0 * total_delay / 100
                - 3 * pdc_count
            )


            next_s_idx = state_to_index(state)
            Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[next_s_idx]) - Q[s_idx, action])

            if verbose:
                print(f"[Ep{ep:4d}][St{step}] ΔPMU={delta_covered}, delay={total_delay:.1f}, PDCs={pdc_count}, R={reward:.2f}")

            # if len(pmu_covered) == len(pmu_nodes): # tutte le PMU sono coperte
            #     break

        # finito l'episodio, calcola il punteggio per quell'episodio ( il Q-value valuta l'action effettuato )
        final_paths = find_best_paths(state)
        total_delay = sum(data["delay"] for data in final_paths.values())
        pdc_count = sum(state)
        latency_avg = total_delay / len(final_paths) if final_paths else float('inf')
        # Conta i PDC attivi ma non usati nei path
        used_pdcs = set()
        for data in final_paths.values():
            path = data["path"]
            for node in path[1:-1]:
                used_pdcs.add(node)

        useless_pdc_count = sum(
            1 for i, b in enumerate(state)
            if b == 1 and nodes_pdc[i] not in used_pdcs
        )

        final_score = (
            1000 * len(final_paths) 
            - 5 * latency_avg 
            - 20 * pdc_count
            - 50 * useless_pdc_count  # penalità forte per sprechi
        )


        #final_score = 1000 * len(final_paths) - total_delay - 20 * pdc_count

        if final_score > best_score:
            best_score = final_score
            best_state = state.copy()

    # Esito finale
    # selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}
    # print("\n Politica appresa:")
    # print(f" Stato binario: {''.join(str(b) for b in best_state)}")
    # print(f" Nodi PDC selezionati: {selected_pdc}")
    # 🔎 Ricava i PDC usati nei cammini finali
    used_pdcs = set()
    for data in final_paths.values():
        path = data["path"]
        for node in path[1:-1]:  # solo nodi intermedi tra PMU e CC
            used_pdcs.add(node)


    # 🧼 Pulisci best_state per mantenere solo i PDC usati
    clean_best_state = [0] * len(nodes_pdc)
    for i, n in enumerate(nodes_pdc):
        if best_state[i] == 1 and n in used_pdcs:
            clean_best_state[i] = 1
    best_state = clean_best_state
    
    selected_pdc = {nodes_pdc[i] for i, b in enumerate(best_state) if b == 1}


    print("\n📡 Cammini PMU → CC (validi con PDC selezionati):")
    final_paths = find_best_paths(best_state)
    for pmu in pmu_nodes:
        if pmu in final_paths:
            path = final_paths[pmu]["path"]
            delay = final_paths[pmu]["delay"]

            status = " OK" if delay <= max_latency else f" Ritardo {delay:.2f} ms > soglia"
            print(f"  {pmu} → CC: {path} | Ritardo = {delay:.2f} ms {status}")
        else:
            print(f"  {pmu} → CC:  Nessun path valido")

    return selected_pdc, final_paths, max_latency if selected_pdc else None






def choose_algorithm(G):
    print("Scegli un algoritmo di posizionamento PDC:")
    print("1. Greedy (con latenza massima)")
    print("2. Random (con numero di PDC specificato)")
    print("3. Q-Learning")
    print("4. GNN + Policy Gradient")
    print("5. Bruteforce")
    print("6. Esci")

    choice = input("Inserisci il numero dell'algoritmo: ")

    if choice == "1":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return place_pdcs_greedy(G, max_latency)
    elif choice == "2":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        seed = int(input("Inserisci il seed (lascia vuoto per nessun seed): ") or 42)
        return place_pdcs_random(G, max_latency, seed)
    elif choice == "3":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return q_learning_placement(G, max_latency)
    elif choice == "4":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        #return train_with_policy_gradient(G, max_latency)
    elif choice == "5":
        max_latency = int(input("Inserisci la latenza massima (in ms): "))
        return place_pdcs_bruteforce(G, max_latency)
    elif choice == "6":
        print("Uscita in corso...")
        exit(0)
    else:
        print("Scelta non valida. Riprova.")
        return choose_algorithm(G)

def main():
    while True:
        G = create_graph(seed=42)
        modify_latency(G)
        modify_edge_status(G)
        (pdcs,path, max_latency) = choose_algorithm(G)
        print("PDC assegnati nei cluster:", pdcs)
        draw_graph(G, pdcs=pdcs,paths=path, max_latency=max_latency)

if __name__ == "__main__":
    main()