import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G, pdcs=None):
    plt.figure()
    if pdcs is None:
        pdcs = set()

    pmu_nodes = [n for n in G.nodes if G.degree[n] == 1 and n != "CC"]

   
    try:
        pos = nx.nx_pydot.pydot_layout(G, prog="dot")
    except:
        print(" Errore con pydot. Uso spring_layout.")
        pos = nx.spring_layout(G, seed=42)

    labels = nx.get_edge_attributes(G, "latency")
    node_colors = []
    node_labels = {}

    for n in G.nodes:
        label = n

        # Colori
        if n in pdcs:
            node_colors.append("orange")
            label += " (PDC)"
        elif G.nodes[n].get("type") == "control":
            node_colors.append("red")
        else:
            node_colors.append("lightblue")

        # Etichette (PMU)
        if n in pmu_nodes:
            label += " (PMU)"

        node_labels[n] = label

    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_size=1000, node_color=node_colors, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Grafo Gerarchico — Rosso: CC, Arancione: PDC, Azzurro: altri nodi", fontsize=10)
    print(" PDC visualizzati in arancione:", pdcs)
    print(" PMU etichettati (nodi foglia):", pmu_nodes)
    plt.show(block=False)

    
    
    