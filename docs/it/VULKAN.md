# Backend Vulkan

[English](../en/VULKAN.md) · **Italiano**

`lightpfn.vulkan` esegue l'inferenza di LightPFN su qualsiasi GPU con un driver Vulkan: AMD, Intel e NVIDIA, su
Linux e Windows. Non servono CUDA, ROCm né l'SDK Vulkan: i kernel di calcolo sono scritti in WGSL e
compilati in SPIR-V da [wgpu](https://github.com/pygfx/wgpu-py) all'avvio del backend.

```bash
pip install "LightPFN[vulkan]"
python -c "import lightpfn.vulkan as v; print(v.adapters())"
```

```python
from lightpfn import LightPFNClassifier

clf = LightPFNClassifier(device="vulkan", n_estimators=4).fit(X_train, y_train)
```

`device="auto"` usa Vulkan quando PyTorch non vede una GPU CUDA o ROCm e wgpu trova una GPU Vulkan.

## Scegliere l'adattatore

`device="vulkan"` prende la prima GPU dedicata, altrimenti una integrata. `device="vulkan:1"` sceglie
l'adattatore con indice 1 in `lightpfn.vulkan.adapters()`. La variabile d'ambiente `LIGHTPFN_VULKAN_ADAPTER`
seleziona un adattatore per indice o per una parte del nome; `LIGHTPFN_VULKAN_ADAPTER=llvmpipe` esegue i kernel
sul driver CPU di Mesa, ed è così che i test girano senza GPU.

## Progetto

Il backend implementa la stessa funzione di `LightPFN.folded()`, la copia della rete per l'inferenza con i pesi
delle normalizzazioni incorporati nelle proiezioni successive, con le stesse due chiamate: `encode` costruisce il
contesto di un training set e `predict_logits` fa passare le righe di test su quel contesto. Le statistiche delle
colonne e la ricerca del rango dei valori (un ordinamento e una ricerca binaria per colonna) girano sull'host in
PyTorch; tutti gli strati successivi girano sulla GPU, e il contesto del training set resta in memoria GPU tra
`fit` e `predict`.

- **Otto kernel.** Un prodotto matriciale a tile con normalizzazione, bias, GELU e residuo opzionali fusi;
  un'attenzione a tile con softmax online (come in flash attention) per sequenze di chiavi lunghe; un kernel di
  attenzione per i pochi token di sintesi e induttori; la normalizzazione per riga; le feature delle celle
  (z-score con clipping morbido, rango della CDF empirica, feature di Fourier delle colonne vicine, indicatori di
  valore mancante); gli embedding rotazionali; la scalatura appresa delle query; una copia con passo.
- **Viste invece di copie.** Ogni kernel legge e scrive i tensori attraverso viste con passo (offset di base,
  lunghezza di riga e passi), così lo stesso kernel legge una colonna di celle, una riga di celle, i primi token
  di ogni riga o un parametro ripetuto, senza trasposizioni né copie.
- **Precisione indipendente dal driver.** Tutto è in float32. Seno, coseno ed erf usano una propria riduzione
  dell'argomento e approssimazioni polinomiali (errore sotto 5e-7), così i risultati non dipendono dalla
  precisione delle funzioni trascendenti di ciascun driver.
- **Lavori GPU brevi.** I driver resettano una GPU il cui lavoro dura troppo (il ring timeout di amdgpu su Linux,
  il TDR di 2 s su Windows). I dispatch pesanti vengono tagliati in fette di workgroup e inviati in lavori di
  circa 5e10 operazioni in virgola mobile, pochi millisecondi ciascuno su una GPU dedicata. Durante lo sviluppo
  un encode di 20.000 righe inviato come lavoro unico ha davvero causato un reset del ring; a fette gira
  normalmente.
- **Memoria.** I pesi vengono caricati una volta per classificatore. Le cache del contesto di un estimatore e i
  buffer di lavoro vengono allocati a ogni fit e riusati tra i blocchi di predizione.

## Concordanza con PyTorch

I test confrontano il backend con il percorso PyTorch su reti casuali e, facoltativamente, su un checkpoint B4 storico, con valori mancanti,
colonne costanti, feature di padding e diverse varianti di architettura: i logit coincidono entro circa 1e-6 e le
probabilità attraverso il classificatore entro 3e-5. I test Vulkan girano sulla GPU e su llvmpipe
(`tests/test_vulkan.py`).

## Velocità

Fit più predict con quattro estimatori dell'architettura rilasciata su tabelle binarie sintetiche, desktop con
Intel i7-13700KF (16 thread) e AMD RX 7900 XT (Linux, driver Mesa RADV, PyTorch 2.13 con ROCm 7.2):

| Righe di training x feature, righe di test | CPU, 16 thread | ROCm (PyTorch) | Vulkan | Vulkan rispetto a CPU |
|---|---:|---:|---:|---:|
| 1.000 x 20, 500 | 0,45 s | 0,057 s | 0,044 s | 10,3x |
| 5.000 x 50, 2.000 | 4,84 s | 0,59 s | 0,50 s | 9,6x |
| 10.000 x 100, 5.000 | 19,8 s | 2,48 s | 2,13 s | 9,3x |
| 20.000 x 50, 5.000 | 29,2 s | 4,76 s | 4,86 s | 6,0x |
| 20.000 x 200, 10.000 | 70,4 s | 10,4 s | 8,68 s | 8,1x |

Su questa GPU Vulkan è veloce quanto PyTorch con ROCm, o di più. Il primo fit di un processo compila anche i
kernel che gli servono.

## Limiti

- **Dimensione dei buffer.** Le celle di training di un estimatore (righe x feature x 64 float) devono stare in
  un buffer della GPU (2 GiB sul driver della RX 7900 XT provato, circa 8,4 milioni di celle, per esempio 84.000 righe
  x 100 feature). Oltre, `device="auto"` ripiega sulla CPU con un avviso e `device="vulkan"` solleva
  `MemoryError`. Abbassa `max_context` per restare sulla GPU.
  Altri buffer del contesto e di lavoro possono imporre un limite inferiore; 8,4 milioni di celle è un limite superiore.
- **Piattaforme.** Linux e Windows. macOS non è supportato (wgpu userebbe Metal, che questo backend non usa).
  Provato su AMD RX 7900 XT (Linux con RADV) e su Mesa llvmpipe; la verifica su Windows è ancora da fare. I driver Vulkan Intel
  e NVIDIA usano lo stesso codice ma non sono stati misurati.
- **Addestramento** non supportato: Vulkan serve per l'inferenza. Con colonne marcate categoriche, l'adapter sperimentale
  (`Config.cat_adapter`, spento nel modello rilasciato) ripiega sulla CPU con `device="auto"`;
  un `device="vulkan"` esplicito solleva `MemoryError`.

## Alternative provate

Abbiamo provato anche due compilatori generici. IREE non aveva una traduzione dell'attenzione per Vulkan,
generava codice sbagliato per RDNA3 su una parte della rete ed era lento con forme dinamiche. Il provider WebGPU
di ONNX Runtime calcolava l'attenzione fusa con errori dal 2 al 17% sulle nostre forme. I kernel scritti a mano
coincidono con PyTorch a meno degli arrotondamenti e sono veloci quanto PyTorch con ROCm sulla stessa GPU.
