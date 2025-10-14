# 🧠 Reinforcement Learning per posizionamento dei PDC – Appunti

Questo documento riassume quanto appreso sull’utilizzo del **Q-learning** per posizionare i PDC (Phasor Data Concentrator) in un grafo rappresentante una rete.

---

## 📌 Obiettivo

Posizionare uno o più PDC su determinati nodi di un grafo in modo da:

- Garantire che tutti i PMU (Phasor Measurement Unit) siano "coperti"
- Rispettare una soglia massima di latenza (`max_latency`)
- Imparare automaticamente una strategia ottimale usando Reinforcement Learning (Q-learning)

---

## ⚙️ Funzionamento del codice

### 1. Ambiente (grafo)
Un grafo `G` creato con NetworkX, dove:

- I nodi rappresentano dispositivi: PMU, PDC, nodo centrale (CC)
- Gli archi hanno un peso chiamato `latency`

### 2. Stati
Ogni stato è rappresentato da un vettore binario, es: `[1, 0, 1]`, che indica **quali nodi hanno un PDC**.

Con 3 nodi candidati (es. `1`, `2`, `3`), abbiamo `2^3 = 8` stati possibili.

### 3. Azioni
Le azioni possibili sono **piazzare un PDC** in uno dei nodi candidati.

### 4. Q-table
È una matrice `8 x 3` (`num_states x num_actions`) che memorizza, per ogni stato, **quanto è utile** compiere ogni azione.

### 5. Reward
- `+10`: se tutti i PMU sono coperti entro la latenza massima
- `-1`: mossa neutra
- `-5`: penalità per aver piazzato un PDC dove già ce n’è uno

### 6. Algoritmo
Si usa il **Q-learning** con strategia epsilon-greedy:
- `epsilon = 0.2`: il 20% delle volte l’agente esplora (sceglie una mossa casuale)
- `alpha = 0.1`: learning rate
- `gamma = 0.9`: fattore di sconto per i reward futuri

---

## 🔎 Funzione chiave: `is_pmu_covered(pdc_nodes)`

Verifica se **ogni PMU** è coperto da un PDC lungo il suo path verso `"CC"`, **senza superare `max_latency`**.

---

## 📊 Interpretare la Q-table

Esempio di Q-table finale:

000 | 7.06 | 10.00 | 9.66
001 | 0.00 | 0.00 | 0.00
010 | 0.00 | 0.00 | 0.00
011 | 0.00 | 0.00 | 0.00
100 | 0.02 | 9.75 | 3.44
101 | 0.00 | 0.00 | 0.00
110 | 0.00 | 0.00 | 0.00
111 | 0.00 | 0.00 | 0.00


### Come leggere:

- Stato `000` → nessun PDC ancora piazzato
- Colonna `Nodo 2` ha Q-value = `10.00` → azione ottimale
- Significa: **piazzare un PDC in nodo 2 è la scelta migliore**

### Stato `100` (cioè PDC già piazzato in nodo 1):

- Aggiungere un PDC anche in nodo `2` (valore `9.75`) è una buona mossa di "recupero"

### Stati con tutti `0.00`
Indicano:
- Nessuna azione utile disponibile
- Oppure lo stato è già terminale o non è mai stato visitato

---

## ✅ Risultato ottenuto

- L’agente ha imparato che **piazzare un solo PDC nel nodo 2** è sufficiente per soddisfare i vincoli
- Lo stato iniziale (`000`) suggerisce correttamente di piazzare il PDC in nodo `2`
- La strategia è efficiente e compatta

---

## 📌 Prossimi passi

- Estendere il grafo con più PMU o nodi
- Migliorare la funzione di reward per includere altri obiettivi (es. minimizzare il numero totale di PDC)
- Passare a **Deep Q-learning** (DQN) se i nodi diventano troppi per la Q-table
