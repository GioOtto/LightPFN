# Backend Vulkan

La versione 1.5 supporta su Vulkan la classificazione con adattatore categorico e la regressione quantile.
Entrambi gli estimatori usano fp32 su questo backend; `precision="fp16"` richiede CUDA/ROCm.

[English](../en/VULKAN.md) · **Italiano**

`lightpfn.vulkan` esegue l'inferenza di LightPFN su qualsiasi GPU con un driver Vulkan: AMD, Intel e NVIDIA, su
Linux e Windows. Non servono CUDA, ROCm né l'SDK Vulkan: i kernel di calcolo sono scritti in WGSL e
compilati in SPIR-V da [wgpu](https://github.com/pygfx/wgpu-py) all'avvio del backend.

```bash
pip install "LightPFN[vulkan]"
python -c "import lightpfn.vulkan as v; print(v.adapters())"
```

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor

clf = LightPFNClassifier(device="vulkan").fit(X_train, y_class)
reg = LightPFNRegressor(device="vulkan").fit(X_train, y_reg)
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

Per entrambi gli estimatori, `device="auto"` prova CUDA/ROCm tramite PyTorch, poi Vulkan, poi CPU. La variabile
d'ambiente `LIGHTPFN_DEVICE` sostituisce `"auto"`. Vulkan viene scelto quando PyTorch non vede una GPU CUDA/ROCm
e wgpu trova una GPU Vulkan. `to()` sposta solo tra dispositivi torch; per passare a Vulkan serve un nuovo fit.

## Scegliere l'adattatore

`device="vulkan"` prende la prima GPU dedicata, altrimenti una integrata. `device="vulkan:1"` sceglie
l'adattatore con indice 1 in `lightpfn.vulkan.adapters()`. La variabile d'ambiente `LIGHTPFN_VULKAN_ADAPTER`
seleziona un adattatore per indice o per una parte del nome; `LIGHTPFN_VULKAN_ADAPTER=llvmpipe` esegue i kernel
sul driver CPU di Mesa, ed è così che i test girano senza GPU.

## Progetto

Il backend implementa lo stesso calcolo della rete di inferenza con normalizzazioni incorporate: `encode`
costruisce il contesto, `predict_logits` gestisce la classificazione e `predict_quantiles` la regressione.

Embedding delle celle (incluse le feature categoriche di Fourier), proiezioni delle statistiche categoriche
(`CatStatsEmbed`), stadi di colonna, raffinamento delle righe, stadio di riga, blocchi transformer ICL e attenzione
del decoder girano sulla GPU. Statistiche delle colonne, normalizzazione dei valori e ricerca del rango
(ordinamento/ricerca), statistiche categoriche (`context_stats` e `query_stats`) e preparazione dei loro pesi di
proiezione girano sull'host in PyTorch. Anche la normalizzazione del target di regressione gira sull'host.
Dopo che l'attenzione del decoder GPU calcola i momenti, la piccola lettura finale della testa quantile (incluse
normalizzazione di riga e media/scala dei momenti), il ritorno alle unità del target, la media dei membri e
l'integrazione della predizione puntuale girano sull'host. Il ritorno alle unità del target usa float64. Le
cache di colonna, chiavi/valori ICL e chiavi del decoder restano in memoria GPU tra `fit` e `predict`.

- **Otto kernel.** Un prodotto matriciale a tile con normalizzazione, bias, GELU e residuo opzionali fusi;
  un'attenzione a tile con softmax online (come in flash attention) per sequenze di chiavi lunghe; un kernel di
  attenzione per i pochi token di sintesi e induttori; la normalizzazione per riga; le feature delle celle
  (z-score con clipping morbido, rango della CDF empirica, feature di Fourier delle colonne vicine, indicatori di
  valore mancante); gli embedding rotazionali; la scalatura appresa delle query; una copia con passo.
- **Viste invece di copie.** Ogni kernel legge e scrive i tensori attraverso viste con passo (offset di base,
  lunghezza di riga e passi), così lo stesso kernel legge una colonna di celle, una riga di celle, i primi token
  di ogni riga o un parametro ripetuto, senza trasposizioni né copie.
- **Precisione indipendente dal driver.** Tutti i kernel Vulkan usano float32. Seno, coseno ed erf usano una propria riduzione
  dell'argomento e approssimazioni polinomiali (errore sotto 5e-7), così i risultati non dipendono dalla
  precisione delle funzioni trascendenti di ciascun driver.
- **Lavori GPU brevi.** I driver resettano una GPU il cui lavoro dura troppo (il ring timeout di amdgpu su Linux,
  il TDR di 2 s su Windows). I dispatch pesanti vengono tagliati in fette di workgroup e inviati in lavori di
  circa 5e10 operazioni in virgola mobile, pochi millisecondi ciascuno su una GPU dedicata. Durante lo sviluppo
  un encode di 20.000 righe inviato come lavoro unico ha davvero causato un reset del ring; a fette gira
  normalmente.
- **Memoria.** I pesi vengono caricati una volta per modello. Le cache del contesto di un estimatore e i
  buffer di lavoro vengono allocati a ogni fit e riusati tra i blocchi di predizione.

## Concordanza con PyTorch

I test confrontano il backend con torch CPU fp32 su reti casuali, un checkpoint B4 storico opzionale e i
checkpoint finali 1.5 di classificazione/regressione. Coprono categorie mancanti e mai viste, classi assenti,
colonne costanti, feature di padding, entrambi i membri di regressione e il ripiego per limiti dei buffer. I
test Vulkan girano anche su Mesa llvmpipe senza GPU (`tests/test_vulkan.py`).

Su una RX 7900 XT (Linux, Mesa RADV), il classificatore finale 1.5 è stato verificato su tabelle miste con tre
colonne categoriche (5, 60 e 400 livelli, 5% di valori mancanti), con 600, 4.000 e 12.000 righe di contesto.
Le probabilità coincidono con torch CPU entro 2.6e-5 in differenza assoluta, con argmax identico su tutte le
righe verificate. Il regressore a due membri è stato verificato su cinque sorgenti di regressione fino a
8.000 righe di contesto, con differenze relative dei quantili al massimo 9e-6. I controlli fino a 40.000 righe
di contesto hanno raggiunto al massimo 5.5e-5 di differenza relativa sui quantili e 3.4e-5 sulle predizioni
puntuali; RMSE invariato alla precisione riportata. Sono concordanze misurate sulle tabelle verificate, non
limiti universali dell'errore.

## Velocità

Gli stessi controlli 1.5 sulla RX 7900 XT hanno misurato fit più predict CPU contro Vulkan di 0,36 contro
0,53 s a 600 righe di contesto, 1,86 contro 0,26 s a 4.000 e 8,24 contro 1,56 s a 12.000 in classificazione.
La regressione a 40.000 righe di contesto era circa 3 volte più veloce della CPU. I casi con contesti più
grandi erano da 3 a 7 volte più veloci; la tabella piccola di classificazione era più lenta su Vulkan. I driver
Vulkan Intel e NVIDIA usano lo stesso codice ma non sono stati misurati. Il primo fit di un processo compila
anche i kernel che gli servono.

Per confronto storico, le misure seguenti riguardano la **1.1 senza adattatore categorico**, con quattro
estimatori su tabelle binarie sintetiche: Intel i7-13700KF (16 thread), AMD RX 7900 XT, Linux, Mesa RADV,
PyTorch 2.13 con ROCm 7.2. Non sono misure della 1.5.

| Righe di training x feature, righe di test | CPU, 16 thread | ROCm (PyTorch) | Vulkan | Vulkan rispetto a CPU |
|---|---:|---:|---:|---:|
| 1.000 x 20, 500 | 0,45 s | 0,057 s | 0,044 s | 10,3x |
| 5.000 x 50, 2.000 | 4,84 s | 0,59 s | 0,50 s | 9,6x |
| 10.000 x 100, 5.000 | 19,8 s | 2,48 s | 2,13 s | 9,3x |
| 20.000 x 50, 5.000 | 29,2 s | 4,76 s | 4,86 s | 6,0x |
| 20.000 x 200, 10.000 | 70,4 s | 10,4 s | 8,68 s | 8,1x |

## Limiti

- **Dimensione dei buffer.** Le celle di training di un estimatore (righe x feature x 64 float) devono stare in
  un buffer della GPU (2 GiB sul driver della RX 7900 XT provato, circa 8,4 milioni di celle, per esempio 84.000 righe
  x 100 feature). Oltre, `device="auto"` ripiega sulla CPU con un avviso e `device="vulkan"` solleva
  `MemoryError`. Abbassa `max_context` per restare sulla GPU.
  Questa regola vale per classificazione e regressione. Altri buffer del contesto e di lavoro, inclusa la cache
  di regressione a due membri, possono imporre un limite inferiore; 8,4 milioni di celle è un limite superiore.
- **Piattaforme.** Linux e Windows. macOS non è supportato (wgpu userebbe Metal, che questo backend non usa).
  Provato su AMD RX 7900 XT (Linux con RADV) e su Mesa llvmpipe; la verifica su Windows è ancora da fare. I driver Vulkan Intel
  e NVIDIA usano lo stesso codice ma non sono stati misurati.
- **Addestramento** non supportato: Vulkan serve per l'inferenza, inclusi adattatore categorico e regressione.

## Alternative provate

Abbiamo provato anche due compilatori generici. IREE non aveva una traduzione dell'attenzione per Vulkan,
generava codice sbagliato per RDNA3 su una parte della rete ed era lento con forme dinamiche. Il provider WebGPU
di ONNX Runtime calcolava l'attenzione fusa con errori dal 2 al 17% sulle nostre forme. I kernel scritti a mano
coincidono con PyTorch a meno degli arrotondamenti e sono veloci quanto PyTorch con ROCm sulla stessa GPU.
