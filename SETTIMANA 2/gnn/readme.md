# Graph Neural Network (GNN) per PDC Placement

## 🧩 1. Ambiente (PDCEnv)

- Gestisce il grafo, i vincoli, i reward
- Fornisce uno stato e accetta azioni

## 🧩 2. Grafo come input: torch_geometric.data.Data
Per poter dare un grafo a una GNN, serve:

- **`x`**: feature di ogni nodo (es. è PMU? è CC? è PDC?)
- **`edge_index`**: tensore `[2, num_edges]` con tutti gli archi
- **`edge_attr`**: latenza degli archi (se vuoi usarla nella GNN)

> 👉 Questo lo crei da networkx con `torch_geometric.utils.from_networkx` oppure manualmente.

## 🧩 3. GNN (Graph Neural Network)
- Rete neurale che riceve il grafo come input
- Produce un **embedding** (vettore) per ogni nodo → rappresenta lo stato locale del nodo in base alla struttura globale
- **Esempi**: GCN, GraphSAGE, GAT (fornite da `torch_geometric.nn`)

## 🧩 4. Policy Network (Actor)
- Prende gli embedding dei nodi
- Ritorna una distribuzione di probabilità sui nodi validi
- Sceglie (campiona) un nodo da attivare come PDC
- Questa rete viene allenata con un algoritmo RL

## 🧩 5. Algoritmo di RL
Noi usiamo **Policy Gradient** (es. REINFORCE) per semplicità

**L'idea è:**
1. **Esplorare** → provare azioni → raccogliere reward
2. **Aggiornare** la rete per massimizzare il reward futuro atteso -> Un gradiente è un vettore che indica la direzione e la quantità con cui modificare un peso di un modello per migliorare la performance (cioè per ridurre la loss, la funzione di errore). È la derivata della funzione di perdita (loss) rispetto a un peso, quindi "Quanto cambia l’errore (loss) se cambio leggermente questo peso?". Infatti, quando chiamo "loss.backward()" , ogni peso w del tuo modello ha associato un valore chiamato w.grad. Questo valore è il gradiente della loss. Se nel training un nodo viene scelto spesso ma porta a un reward negativo, allora i gradienti diranno: "Abbassa la probabilità associata a quel nodo", modificando i pesi in modo da produrre logits (uscite grezze del modello) diversi rispetto a quel peso. Quindi, in sintesi,quando chiami loss.backward(), vengono calcolati **i gradienti della loss rispetto a tutti i pesi del modello, cioè quei parametri interni che trasformano le feature e determinano le probabilità**

---

## 🔁 Ciclo di training tipico (REINFORCE semplificato)

1. **Resetta** l'ambiente
2. **Fai una lista vuota** di `(state, action, reward)`

**Per ogni step:**
1. Converte il grafo in `Data`
2. GNN → embedding dei nodi
3. Policy → distribuzione softmax su nodi validi
4. Campiona un nodo
5. `env.step(node)` → ottieni reward
6. Salva `(state, action, reward)`

**A fine episodio:**
1. Calcola reward cumulativo
2. Calcola perdita negativa: `loss = -log(prob(action)) * reward`
3. Backpropagation + ottimizzazione

---

## 📚 Concetti Fondamentali

### Cos'è un tensore, cosa si intende per softmax e cosa sono gli embedding?

### 🔢 Tensore
Un tensore è una struttura dati per contenere numeri, come array e matrici, ma generalizzata:

| Tipo di dato | In matematica | In PyTorch (tensor shape) |
|--------------|---------------|---------------------------|
| Singolo numero | Scalare | `()` → es. `torch.tensor(7)` |
| Lista di numeri | Vettore | `(n,)` → es. `torch.tensor([1, 2, 3])` |
| Tabella (righe x colonne) | Matrice | `(n, m)` → es. `torch.tensor([[1, 2], [3, 4]])` |
| Serie di matrici | Tensore 3D+ | `(batch, n, m)` → es. immagini, grafi |

**In pratica:**
- I dati dei nodi, come "è PMU / CC / normale", sono tensori 1D
- Gli embedding prodotti dalla GNN per ogni nodo sono tensori 2D: uno per nodo

### 🎯 Softmax
La softmax è una funzione matematica che:

1. **Prende** un vettore di numeri reali (es. output della rete neurale)
2. **Li trasforma** in una distribuzione di probabilità:
   - Tutti positivi
   - Somma = 1
   - Valori alti diventano probabilità più alte

**In questo caso:**
- L'actor (rete di policy) produce un vettore con un punteggio per ogni nodo
- La softmax trasforma questi punteggi in probabilità su quale nodo scegliere
- L'agente campiona (sceglie a caso, ma guidato dalle probabilità)

### 🎯 Embedding

Un **embedding** è una rappresentazione numerica densa e vettoriale di un oggetto complesso.

**Nel tuo caso:**
- Ogni nodo del grafo ha delle caratteristiche discrete o simboliche (es. "è una PMU?", "è il CC?", "è un nodo normale?")
- La GNN trasforma queste informazioni in un vettore di numeri → chiamato **embedding del nodo**
- L'actor (rete di policy) usa gli embedding per decidere quale nodo scegliere come PDC
- L'embedding dice alla rete: "questo nodo è vicino a PMU?", "è utile?", "è centrale?", ecc.

**Metafora:**

Immagina una GNN come un traduttore:
- **Tu dai in input**: "nodo 4 è un nodo normale con certi vicini"
- **Lei ti dà in output**: un vettore numerico (embedding) che riassume tutto quello che c'è da sapere su quel nodo nella rete
