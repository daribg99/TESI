import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

def draw_graph(G, pdcs=None):
    if pdcs is None:
        pdcs = set()

    plt.figure(figsize=(14, 10))

    # Posizioni dei nodi: layout gerarchico se disponibile
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

        # Colore e descrizione in base al ruolo
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

    # Disegna nodi
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           edgecolors=node_edgecolors,
                           node_size=1100,
                           linewidths=1.8)

    # Disegna archi e etichette
    nx.draw_networkx_edges(G, pos, width=1.2)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

    # Legenda
    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="CC"),
        Patch(facecolor="lightgreen", edgecolor="black", label="PMU"),
        Patch(facecolor="orange", edgecolor="black", label="PDC (selezionati)"),
        Patch(facecolor="lightblue", edgecolor="gray", label="Altro nodo (candidato)")
    ]
    plt.legend(handles=legend_elements, loc="lower left", fontsize=9, frameon=True)

    plt.title("Grafo con ruoli: CC (rosso), PMU (verde), PDC (arancione), altri (azzurro)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)
