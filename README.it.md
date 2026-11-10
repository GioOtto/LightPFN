<div align="center">
  <img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/logo.svg" width="112" height="112" alt="Logo di LightPFN: una piccola tabella con una cella evidenziata" />
  <h1>LightPFN</h1>
  <p><strong>Modelli fondazionali tabellari compatti da circa 4,7 milioni di parametri per classificazione e regressione.</strong></p>
  <p>Preaddestrati solo su dati sintetici. Estimatori scikit-learn che girano su CPU, CUDA, ROCm e GPU Vulkan.</p>

  <p>
    <a href="https://pypi.org/project/LightPFN/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lightpfn?style=for-the-badge&color=111111&label=PyPI" /></a>
    <a href="https://huggingface.co/ueuegio/LightPFN"><img alt="Pesi su Hugging Face" src="https://img.shields.io/badge/Pesi-Hugging%20Face-111111?style=for-the-badge&logo=huggingface&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf"><img alt="Report tecnico 1.5 (PDF, in inglese)" src="https://img.shields.io/badge/Report%201.5-PDF-8B1A1A?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf"><img alt="Report della 1.1 (PDF, in inglese)" src="https://img.shields.io/badge/Report%201.1-PDF-444444?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
  </p>

  <p>
    <a href="docs/it/GUIDA.md">Guida</a> &nbsp;&nbsp;
    <a href="docs/it/RISULTATI.md">Risultati</a> &nbsp;&nbsp;
    <a href="docs/it/VULKAN.md">Backend Vulkan</a> &nbsp;&nbsp;
    <a href="docs/it/ADDESTRAMENTO.md">Addestramento</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/issues">Segnalazioni</a>
  </p>
  <p><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-181818" /> <img alt="4,6M / 4,8M parametri" src="https://img.shields.io/badge/parametri-4.6M%20%7C%204.8M-181818" /> <img alt="Codice e pesi Apache 2.0" src="https://img.shields.io/badge/codice%20%2B%20pesi-Apache%202.0-181818" /> <img alt="Addestrato solo su dati sintetici" src="https://img.shields.io/badge/dati%20di%20training-solo%20sintetici-181818" /></p>
  <p><a href="README.md">English</a> &nbsp; <strong>Italiano</strong></p>
</div>

## Che cos'è

LightPFN è una famiglia di prior-data fitted network: transformer preaddestrati una volta sola su milioni di
problemi tabellari sintetici. Con `fit` memorizzano e codificano il training set come contesto; con
`predict_proba`, `predict` o `predict_quantiles` leggono contesto e righe di test in un solo passaggio in
avanti. Non ottimizzano i pesi sui tuoi dati e non richiedono tuning degli iperparametri.

La versione 1.5 fornisce due checkpoint separati per classificazione (da 2 a 10 classi) e regressione:

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor

clf = LightPFNClassifier().fit(X_train, y_train_cls)
proba = clf.predict_proba(X_test)

reg = LightPFNRegressor().fit(X_train, y_train_reg)
pred = reg.predict(X_test)
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

- **Piccolo.** 4.609.472 parametri per il classificatore 1.5 e 4.772.519 per il regressore 1.5 (circa 19 MB di
  pesi ciascuno; 4.603.088 parametri, 18 MB nella 1.1), pensati per girare sia su una CPU comune sia su GPU.
  TabICLv2 ne ha sei volte tanti.
- **Accurato senza tuning.** Nella classifica ufficiale di TabArena (38 dataset di classificazione),
  LightPFN 1.1 all'11 ottobre 2026 era 21° su 92 metodi, sopra tutti i metodi ad alberi, compresi LightGBM,
  CatBoost e XGBoost ottimizzati e in ensemble; gli intervalli dei più vicini si sovrappongono al suo.
  LightPFN 1.5 non ha ancora una voce nella classifica ufficiale.
- **Qualsiasi GPU.** CUDA e ROCm tramite PyTorch (con inferenza fp16 automatica nella 1.5), più un backend
  Vulkan con kernel di calcolo WGSL propri per GPU AMD, Intel e NVIDIA su Linux e Windows, senza installare
  CUDA o ROCm. Vulkan supporta l'adattatore categorico e la regressione. Su una RX 7900 XT, i casi 1.5
  misurati con contesti più grandi sono da 3 a 7 volte più veloci della CPU; le tabelle piccole possono essere
  più lente. I driver Vulkan Intel e NVIDIA usano lo stesso codice ma non sono stati misurati. Vedi
  [Dispositivi e precisione](#dispositivi-e-precisione).
- **API scikit-learn.** `Pipeline`, `GridSearchCV`, `clone`, `cross_val_score` e DataFrame pandas con colonne
  categoriche, stringhe, booleane e valori mancanti, più il rilevamento automatico `cat="auto"`.
- **Aperto.** Codice, pesi e l'intera pipeline di addestramento sotto Apache 2.0. Addestrato da zero solo su
  dati sintetici: niente distillazione, niente pesi o output di altri modelli fondazionali tabellari.

## Novità della 1.5

- Adattatore categorico addestrato da zero e rilevamento `cat="auto"` (dtype pandas categorici, stringhe,
  object e bool, più colonne numeriche intere con da 2 a 20 livelli distinti).
- `LightPFNRegressor`: media predittiva e 999 quantili (da `.001` a `.999`), due membri e contesto fino a
  50.000 righe di default.
- `precision="auto"`: fp16 su GPU CUDA/ROCm compatibili sopra la soglia di celle (2.048 celle di contesto per
  membro), fp32 su CPU. I blocchi non finiti in fp16 vengono rieseguiti automaticamente in fp32.
- Stadio di addestramento a contesto lungo con tabelle fino a 60.000 righe per entrambi i modelli.
- Classificatore affinato su tabelle sintetiche con 64-500 feature (il preaddestramento si fermava a 100):
  migliore sulle tabelle larghe, invariato altrove entro il rumore.
- Directory separate `classifier/` e `regressor/` (`config.json` e `model.safetensors`) su Hugging Face
  (`ueuegio/LightPFN`), fissate per commit.

## Benchmark (LightPFN 1.1)

**[TabArena](https://huggingface.co/spaces/TabArena/leaderboard), classifica ufficiale, 38 dataset di
classificazione (LightPFN 1.1).** I maintainer di TabArena hanno eseguito LightPFN 1.1 nella configurazione di
default (quattro estimatori) e ne hanno verificato il risultato. LightPFN 1.5 non è ancora stato valutato nella
classifica ufficiale. Classifica dell'11 ottobre 2026, modelli imputati esclusi:

| Vista (LightPFN 1.1) | Posizione | Elo (IC 95%) |
|---|---:|---:|
| Tutti gli split | **21° su 92** | **1441** (+69 / -79) |
| TabArena-Lite (solo il primo split) | 20° su 92 | 1426 (+71 / -70) |

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_elo.png" width="100%" alt="TabArena, 38 dataset di classificazione, tutti gli split: Elo con intervalli al 95% per una selezione di metodi; LightPFN 1.1 è 21°, sopra tutti i metodi ad alberi" /></p>

- **Sopra tutti i metodi ad alberi**, anche ottimizzati e in ensemble. Gli intervalli si sovrappongono a quelli
  di CatBoost di default e delle versioni ottimizzate dei gradient boosting, quindi contro di loro non è una
  vittoria netta; XGBoost e LightGBM di default sono più di 200 punti Elo sotto.
- **Dietro ai modelli fondazionali più grandi.** I primi 18 posti sono tutti modelli fondazionali; TabICLv2 è
  avanti di 134 punti.
- **Per dimensione:** 19° sui 28 dataset piccoli, sopra tutti i metodi ad alberi; 44° su 93 sui 10 medi, dove
  CatBoost di default è 17°.

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_pareto.png" width="100%" alt="TabArena: Elo rispetto al tempo mediano di addestramento per 92 metodi; LightPFN 1.1 è sul fronte di Pareto" /></p>

LightPFN 1.1 è sul fronte di Pareto tra Elo e tempo. Con 0,26 s per 1.000 righe solo KNN si addestra più in
fretta; con 0,029 s per 1.000 righe solo EBM e CatBoost di default predicono più in fretta, entrambi con un Elo
più basso. I tempi sono quelli di TabArena, con LightPFN 1.1 su GPU.

Per avere questa vista della voce 1.1 nella pagina della classifica, seleziona **Classification** e togli la
spunta a **Include imputed models**. La vista di default comprende anche i 13 dataset di regressione, che
LightPFN 1.1 non supportava, quindi lì vengono imputati.
[Protocollo, CSV e la valutazione precedente degli autori](docs/it/RISULTATI.md#tabarena-classifica-ufficiale).

## Installazione

```bash
pip install LightPFN             # CPU, CUDA o ROCm, tramite il PyTorch installato (lightpfn==1.5.0)
pip install "LightPFN[vulkan]"   # aggiunge il backend GPU Vulkan (wgpu)
```

Serve Python 3.10 o successivo. Il primo `fit` scarica solo il checkpoint necessario (circa 19 MB) da
[Hugging Face](https://huggingface.co/ueuegio/LightPFN) alla revisione fissata dal pacchetto e lo tiene in
cache. Le directory `classifier/` e `regressor/` contengono ciascuna `config.json` e `model.safetensors`; è
possibile usare una directory locale con `checkpoint=`. Su una macchina senza GPU conviene installare prima
PyTorch dall'indice CPU, così non scarica le librerie CUDA:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Primi passi

### Classificazione

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = LightPFNClassifier(random_state=0)
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

Un DataFrame pandas con categoriche e valori mancanti si passa così com'è (pandas è opzionale e serve solo per
gli input DataFrame):

```python
import pandas as pd

df = pd.DataFrame({
    "age": [34, 51, None, 28, 45, 39],
    "plan": ["basic", "pro", "pro", None, "basic", "enterprise"],
    "region": pd.Categorical(["north", "south", "south", "east", "north", "east"]),
    "active": [True, False, True, True, False, True],
})
y = [0, 1, 1, 0, 1, 0]
clf = LightPFNClassifier().fit(df, y)
clf.predict_proba(df.head(2))
```

Le colonne di tipo category, stringa, object o bool diventano codici ordinali delle categorie viste in `fit`. Un
dtype category conserva i livelli dichiarati, anche quelli inutilizzati. I valori mancanti e quelli fuori da quel
vocabolario diventano NaN, che il modello gestisce nativamente. Con il default `cat="auto"`, `fit` marca anche
le colonne numeriche intere che hanno da 2 a 20 livelli distinti nel training set, così vengono lette
dall'adattatore categorico.

### Regressione

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = LightPFNRegressor(random_state=0).fit(X_train, y_train)
pred = reg.predict(X_test)
interval = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

`fit` codifica il contesto senza ottimizzare i pesi. Il classificatore usa di default quattro membri e al
massimo 20.000 righe di contesto per membro; il regressore usa due membri e al massimo 50.000 righe. Altri
esempi in [examples/](examples/) e nella [guida](docs/it/GUIDA.md).

## Dispositivi e precisione

| `device=` | Dove gira il classificatore | Dove gira il regressore |
|---|---|---|
| `"auto"` (predefinito) | una GPU CUDA o ROCm tramite PyTorch, altrimenti una GPU Vulkan, altrimenti la CPU | una GPU CUDA o ROCm tramite PyTorch, altrimenti una GPU Vulkan, altrimenti la CPU |
| `"cpu"` | la CPU, tramite PyTorch | la CPU, tramite PyTorch |
| `"cuda"`, `"cuda:1"` | una GPU NVIDIA (CUDA) o AMD (ROCm) tramite PyTorch | una GPU NVIDIA (CUDA) o AMD (ROCm) tramite PyTorch |
| `"vulkan"`, `"vulkan:1"` | qualsiasi GPU con driver Vulkan, tramite `lightpfn.vulkan`, anche con categoriche marcate | lo stesso backend Vulkan, anche con categoriche marcate |

La variabile d'ambiente `LIGHTPFN_DEVICE` sostituisce `"auto"`.

- **Vulkan nella 1.5.** Entrambi gli estimatori supportano colonne numeriche e categoriche marcate in fp32.
  Embedding delle celle, proiezioni delle statistiche categoriche, stadi di colonna e riga, ICL e attenzione del
  decoder girano sulla GPU. Statistiche di colonne/categorie e ricerca del rango restano sull'host; anche la
  normalizzazione del target di regressione e la lettura finale della testa quantile girano sull'host. Su una
  RX 7900 XT, le probabilità di classificazione hanno una differenza assoluta massima di circa 2.6e-5 da
  torch CPU, con argmax identico, su contesti fino a 12.000 righe con categoriche ad alta cardinalità.
  Le sonde di regressione sui quantili 0.1/0.5/0.9, fino a 40.000 righe, danno differenza assoluta massima /
  quantile CPU massimo in valore assoluto di circa 5.5e-5: non è un limite relativo per ogni quantile,
  e le sonde usavano la regola precedente del secondo membro. Se un fit supera i limiti dei buffer GPU, `device="auto"`
  ripiega sulla CPU con un avviso; `device="vulkan"` esplicito solleva `MemoryError`. `to()` sposta solo tra
  dispositivi torch; per passare a Vulkan serve un nuovo fit. Installa il backend opzionale con
  `pip install "LightPFN[vulkan]"`. Vedi il [backend Vulkan](docs/it/VULKAN.md).
- **Precisione.** `precision="auto"` (predefinito) usa fp16 su GPU CUDA/ROCm compatibili sopra la soglia di celle
  (2.048 celle di contesto per membro) e fp32 su CPU e Vulkan. I blocchi non finiti in fp16 vengono rieseguiti
  automaticamente in fp32. `precision="fp32"` forza float32, con buon accordo sulle configurazioni verificate;
  la differenza Vulkan misurata sopra non è un limite di errore universale. `precision="fp16"` richiede
  CUDA/ROCm. I risultati in fp16 possono differire da quelli in fp32.

## Quando usarlo

LightPFN è una buona scelta predefinita per tabelle di classificazione da 2 a 10 classi e di regressione a
singolo target fino a decine di migliaia di righe, quando serve un modello forte senza tuning degli
iperparametri. Ha questi limiti:

- **Tipi di problema.** La classificazione supporta da 2 a 10 classi; `LightPFNRegressor` supporta la
  regressione a singolo target (media predittiva e quantili). La regressione multioutput non è supportata.
- **Tabelle grandi.** Oltre `max_context` righe (20.000 di default per membro con sottocampionamento
  stratificato in classificazione; 50.000 con sottocampionamento uniforme in regressione), il contesto viene
  sottocampionato. Entrambi i modelli 1.5 includono uno stadio di addestramento a contesto lungo fino a 60.000
  righe. Nel report della 1.1, con tutto il training set come contesto, l'AUC media in classificazione è alla
  pari con CatBoost di default fino a 100.000 righe. Contesti più grandi costano di più in tempo e memoria;
  dimensioni oltre 100.000 righe non sono state valutate.
- **Colonne categoriche e regressione.** La versione 1.5 addestra da zero un adattatore categorico e attiva
  `cat="auto"`. Le tabelle difficili dominate da categoriche ad alta cardinalità e la regressione restano
  comunque dietro ai modelli fondazionali più grandi e, su alcuni task, a CatBoost.
- **Il tempo su CPU cresce con il contesto.** Su TabArena (1.1), quattro estimatori di classificazione impiegano
  una mediana di 1 s per split sulle tabelle piccole, 15 s sulle medie e 54 s sulle grandi, contro 3, 4 e 7 s di
  CatBoost. Se il tempo conta, usa una GPU o `n_estimators=1` (circa quattro volte più veloce in
  classificazione, poco meno accurato).

## Risultati

Tabelle complete, risultati per task, protocolli e tempi: [docs/it/RISULTATI.md](docs/it/RISULTATI.md).

### Risultati locali della 1.5

Le misure della 1.5 sono esecuzioni locali, non una classifica ufficiale.

- **TabArena-Lite (esecuzioni dell'autore, 9 novembre 2026):** 38 task di classificazione, Elo **1436**
  (+74 / -65; 1.1: 1423) con il classificatore finale su una RTX 5090. Sulla stessa GPU il
  classificatore 1.5 precedente al fine-tuning sulle tabelle larghe ha 1416 (1415 sulla RX 7900 XT); quello
  finale ha errore minore su 24 task e maggiore su 14. 13 task di regressione, Elo **1450** (+107 / -103;
  CatBoost default: **1416**), su RX 7900 XT. Sono risultati sul primo split, confrontati con uno snapshot
  delle baseline pubblicate, non risultati verificati dai manutentori. Gli intervalli di Elo del
  classificatore finale, di quello precedente e della 1.1 si sovrappongono. Il regressore usa la regola finale a due membri:
  `log(y)` solo per target positivi con asimmetria campionaria corretta almeno 2 (`log_skew=2.0`),
  altrimenti `-y`. Su CPU fp32 la regola riduce l'RMSE medio relativo per task dello 0,70% su D3-reg e
  dell'1,78% su D3-reg-L rispetto alla regola precedente sulla stessa CPU. La soglia è stata scelta su
  questi set di sviluppo, quindi i miglioramenti stimati sono ottimistici.
- **Velocità di inferenza su GPU (benchmark locale su RTX 5090):** su otto task di classificazione con il
  protocollo 8 fold x 4 membri, il percorso misurato `precision="auto"` dà accelerazioni medie geometriche
  di **1,89x** in train e **2,04x** in predict rispetto alla 1.1.0 sulla stessa GPU (soglia misurata di 8.192
  celle; scalando i tempi ufficiali della 1.1.0 si ottiene circa 0,14 s in train e circa 0,014 s in predict per 1.000 righe; sono rapporti locali, non tempi ufficiali della 1.5).
- **Fine-tuning sulle tabelle larghe (classificatore finale):** il pretraining non è mai andato oltre 100
  feature, e le tabelle con più di 60-100 feature erano l'unica debolezza ripetuta del classificatore 1.5
  precedente. Il classificatore finale lo prosegue per 8.500 passi con metà dei batch da tabelle sintetiche
  con 64-500 feature e il resto dal mix di dati precedente. Differenze appaiate rispetto al classificatore 1.5
  precedente, quattro membri, intervalli bootstrap al 95%: sui nove dataset di D3 con più di 100 feature AUC
  +1,93 punti [-0,06, +5,82] e log loss -0,048 [-0,108, -0,004]; D3 +0,01 punti [-0,06, +0,09]; Dcat 10k
  +0,07 [+0,00, +0,16]; D4 +0,12 [+0,05, +0,21]; errore relativo sul primo split di TabArena -0,85%
  [-1,83, -0,04], -3,3% sui nove task con più di 60 feature. Il primo split di TabArena è stato osservato
  durante il fine-tuning, quindi non è un test indipendente. Con un estimatore il classificatore finale ha
  AUC 0,9102 su D3 e 0,7082 su Dcat 10k.
- **Set di sviluppo di classificazione (classificatore 1.5 prima del fine-tuning sulle tabelle larghe, un estimatore):** su D3 (55 dataset
  OpenML-CC18 non presenti in TabArena) AUC media 0,9103 contro 0,9109 della 1.1.0 e 0,9023 di CatBoost
  default; sui sei dataset con categoriche ad alta cardinalità (Dcat) 0,7075 a 10.000 righe e 0,7134 a 25.000
  righe, contro 0,7047 / 0,7098 della 1.1.0 e 0,7031 / 0,7118 di CatBoost default; sulle tabelle grandi (D4)
  0,8200 contro 0,8188 (1.1.0) e 0,8189 (CatBoost). Sulle tabelle ordinarie la 1.5 è pari alla 1.1.0; il
  guadagno è sulle colonne categoriche.

### Risultati della 1.1

Baseline con le impostazioni di default. Differenze in punti di AUC (100 volte la differenza di AUC) con
intervalli bootstrap appaiati al 95%. Le tabelle qui sotto riguardano i pesi **LightPFN 1.1**.

**TabArena, 38 task di classificazione (LightPFN 1.1, nostro harness)** (split ufficiali, prima ripetizione,
eseguiti con il nostro harness, che non è il protocollo della classifica ufficiale):

| Modello | AUC media | Rank medio su 7 | Errore minore di CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parametri, GPU) | 0,864 | 1,58 | 87% |
| **LightPFN 1.1, 4 estimatori** | **0,858** | **2,50** | **76%** |
| LightPFN 1.1, 1 estimatore | 0,857 | 3,37 | 68% |
| CatBoost | 0,855 | 3,50 | |
| LightGBM | 0,844 | 5,34 | 5% |
| Random forest | 0,838 | 5,89 | 3% |
| XGBoost | 0,833 | 5,82 | 11% |

**Development set: 55 dataset OpenML-CC18 esterni a TabArena (LightPFN 1.1)** (usato per le scelte di progetto,
quindi la stima è ottimistica; al massimo 1.000 righe e 100 feature, cross-validation a cinque fold, un
estimatore):

| Modello | AUC media | LightPFN 1.1 meno modello |
|---|---:|---:|
| **LightPFN 1.1** | **0,911** | |
| CatBoost | 0,902 | +0,86 [0,42, 1,40] |
| Random forest | 0,893 | +1,76 |
| LightGBM | 0,891 | +2,02 |
| XGBoost (54 dataset: il suo wrapper fallisce su uno) | 0,889 | +2,06 [1,32, 2,94] |

## Da 1.1 a 1.5

La 1.5 cambia i pesi predefiniti di classificazione, attiva l'adattatore categorico con `cat="auto"`, usa
`precision="auto"` su GPU CUDA/ROCm e aggiunge `LightPFNRegressor`. Di conseguenza le predizioni cambiano
rispetto alla 1.1. Per conservare esattamente il pacchetto precedente:

```bash
pip install lightpfn==1.1.0
```

Oppure per caricare i pesi 1.1 nel nuovo classificatore (passando `precision="fp32"` per evitare cambi di
precisione):

```python
clf = LightPFNClassifier(
    revision="bd389ab59a89dd0e05c9ecb7c642c08ee52e9637", precision="fp32"
)
```

La revisione 1.1 resta disponibile su Hugging Face con i file alla radice; non include un modello di
regressione.

## Come funziona

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/architecture.png" width="100%" alt="Architettura di LightPFN: embedding delle celle, due stadi per colonna, raffinamento e compressione delle righe, in-context learning, decoder a recupero" /></p>

Ogni cella viene codificata a partire dal suo valore, dal suo rango nella colonna e da un indicatore di valore
mancante. Due stadi per colonna con attenzione indotta leggono il contesto etichettato di ogni colonna. Nella
1.5, un adattatore categorico aggiunge embedding per livello e statistiche ordinate del target per le colonne
marcate come categoriche. Uno stadio per riga con quattro token di sintesi mescola le feature di ogni riga e la
comprime in un vettore. Sette blocchi di in-context learning fanno sì che ogni riga di test guardi le righe di
training. Nel classificatore, un decoder a recupero trasforma l'attenzione in un voto sulle etichette di
training; nel regressore, il decoder recupera i momenti del target dal contesto e una testa quantile predice 999
quantili ordinati da `.001` a `.999`.

Il preaddestramento usa solo problemi sintetici:

- **Classificatore 1.1:** 7,68 milioni di estrazioni da 4,03 milioni di problemi sintetici distinti (90% dal
  prior a grafi causali strutturali `graph` v3 e 10% dal prior a regole `rule` v1 con XOR, parità, tabelle di
  lookup e alberi poco profondi), seguiti da uno stadio a contesto lungo su tabelle fino a 60.000 righe.
- **Classificatore 1.5:** 120.000 passi da 64 problemi (7,68 milioni di estrazioni: 90% `graph` v4 con code
  pesanti e categoriche ad alta cardinalità quantizzate, 10% `rule` v1, entrambi con `p_binary=0.55` e
  `--cat-mask-drop 0.25`), seguiti da uno stadio a contesto lungo di 15.000 passi su tabelle fino a 60.000
  righe.
- **Regressore 1.5:** affinato a partire dal tronco di classificazione sul prior di regressione `r2` (`graph` v3
  con filtro di apprendibilità, rumore osservativo, code pesanti e target discreti), seguito da uno stadio a
  contesto lungo di 5.000 passi fino a 60.000 righe.

Il [report tecnico di LightPFN 1.5](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf) e
il [report della 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf) (in inglese)
descrivono modelli, prior e protocolli di selezione;
[docs/it/ADDESTRAMENTO.md](docs/it/ADDESTRAMENTO.md) spiega come riprodurre l'addestramento.

## Roadmap

La versione 1.5 ha realizzato i due punti della roadmap della 1.1:

- **Feature categoriche native:** l'adattatore categorico è addestrato da zero con la maschera delle categoriche
  attiva fin dal preaddestramento (`--cat-mask-drop 0.25`) su un mix di prior che contiene colonne categoriche
  ad alta cardinalità (`graph` v4), e `cat="auto"` rileva le colonne categoriche in inferenza. (Nella 1.1, un
  adapter addestrato sopra i pesi v1 congelati non aveva aiutato nella configurazione 1.1 verificata; report della 1.1, Sezione 12.)
- **Regressione:** `LightPFNRegressor` predice la media e 999 quantili con un checkpoint dedicato da 4,77
  milioni di parametri.

Vulkan supporta anche l'adattatore categorico e la regressione nella 1.5. Il prossimo passo è la valutazione
ufficiale della 1.5 su TabArena.

## Struttura della repository

| Percorso | Contenuto |
|---|---|
| `lightpfn/` | il pacchetto: modello (`model/`), classificatore (`sklearn.py`) e regressore (`regressor.py`) scikit-learn, gestione della precisione (`precision.py`), dispositivi (`device.py`), backend Vulkan (`vulkan/`), e il codice di addestramento: prior (`prior/`), trainer (`train.py`), harness di valutazione (`eval/`) |
| `tests/` | test unitari e di equivalenza (CPU; i test Vulkan girano anche su un driver CPU) |
| `examples/` | esempi eseguibili di classificazione, regressione, pandas, pipeline e dispositivi |
| `docs/` | guida, risultati, backend Vulkan e addestramento in inglese (`en/`) e in italiano (`it/`), changelog, licenze di terze parti |
| `paper/` | report tecnici (`LightPFN15_report.pdf` per la 1.5 e `LightPFN_report.pdf` per la 1.1), sorgenti LaTeX, figure e dati dei grafici |
| `CHANGELOG.md` | storico delle versioni alla radice della repository |

Il wheel pubblicato contiene solo il codice di inferenza; per l'addestramento serve il sorgente
(`pip install -e ".[train,eval]"`).

## Contribuire e supporto

Segnalazioni di bug e pull request circoscritte sono benvenute: vedi
[CONTRIBUTING](.github/CONTRIBUTING.md). Le vulnerabilità passano da
[SECURITY](.github/SECURITY.md). Le modifiche tra versioni sono in
[CHANGELOG](CHANGELOG.md).

## Citazione

```bibtex
@techreport{ottoboni2026lightpfn15,
  title  = {{LightPFN} 1.5: Native categoricals, quantile regression, and fast portable inference under 5 million parameters},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf}
}

@techreport{ottoboni2026lightpfn,
  title  = {A Sling Against Giants: {LightPFN}, a 4.6M-parameter tabular in-context classifier designed to stay small},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf}
}
```

La prima voce è il report della 1.5, la seconda il report della 1.1. Vedi il [report tecnico di LightPFN 1.5](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
e il [report della 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf). Il pulsante
"Cite this repository" di GitHub legge [CITATION.cff](CITATION.cff).

## Licenza

Codice e pesi: [Apache License 2.0](LICENSE), con l'avviso di attribuzione in [NOTICE](NOTICE). Le
dipendenze mantengono le proprie licenze, elencate in [THIRD_PARTY_LICENSES](docs/THIRD_PARTY_LICENSES.md).
