<div align="center">
  <img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/logo.svg" width="112" height="112" alt="Logo di LightPFN: una piccola tabella con una cella evidenziata" />
  <h1>LightPFN</h1>
  <p><strong>Un modello fondazionale tabellare da 4,6 milioni di parametri per la classificazione.</strong></p>
  <p>Preaddestrato solo su dati sintetici. Un classificatore scikit-learn che gira su CPU, CUDA, ROCm e qualsiasi GPU Vulkan.</p>

  <p>
    <a href="https://pypi.org/project/LightPFN/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lightpfn?style=for-the-badge&color=111111&label=PyPI" /></a>
    <a href="https://huggingface.co/ueuegio/LightPFN"><img alt="Pesi su Hugging Face" src="https://img.shields.io/badge/Pesi-Hugging%20Face-111111?style=for-the-badge&logo=huggingface&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf"><img alt="Report tecnico (PDF, in inglese)" src="https://img.shields.io/badge/Report%20tecnico-PDF-8B1A1A?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
  </p>

  <p>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/it/GUIDA.md">Guida</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/it/RISULTATI.md">Risultati</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/it/VULKAN.md">Backend Vulkan</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/it/ADDESTRAMENTO.md">Addestramento</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/issues">Segnalazioni</a>
  </p>
  <p><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-181818" /> <img alt="4,6M parametri" src="https://img.shields.io/badge/parametri-4.6M-181818" /> <img alt="Codice e pesi Apache 2.0" src="https://img.shields.io/badge/codice%20%2B%20pesi-Apache%202.0-181818" /> <img alt="Addestrato solo su dati sintetici" src="https://img.shields.io/badge/dati%20di%20training-solo%20sintetici-181818" /></p>
  <p><a href="https://github.com/GioOtto/LightPFN/blob/main/README.md">English</a> &nbsp; <strong>Italiano</strong></p>
</div>

## Che cos'è

LightPFN è una prior-data fitted network: un transformer preaddestrato una volta sola su milioni di problemi
di classificazione sintetici. Con `fit` memorizza il training set come contesto; con `predict_proba` legge
contesto e righe di test in un solo passaggio in avanti. Non si allena sui tuoi dati e non richiede tuning.

```python
from lightpfn import LightPFNClassifier

clf = LightPFNClassifier(n_estimators=4).fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

- **Piccolo.** 4.603.088 parametri, 18 MB di pesi, pensato per una CPU comune. TabICLv2 ne ha sei volte
  tanti.
- **Accurato senza tuning.** Su 55 dataset OpenML esterni a TabArena, AUC media 0,911: 0,86 punti sopra
  CatBoost di default e da 1,8 a 2,1 punti sopra LightGBM, XGBoost e random forest di default. Sui 38 task
  di classificazione di TabArena ha un errore minore di CatBoost di default sul 76% dei task.
- **Qualsiasi GPU.** CUDA e ROCm tramite PyTorch, più un backend Vulkan con kernel di calcolo WGSL propri per
  GPU AMD, Intel e NVIDIA su Linux e Windows, senza installare CUDA o ROCm. Su una RX 7900 XT è da 6 a 10
  volte più veloce di una CPU a 16 thread.
- **API scikit-learn.** `Pipeline`, `GridSearchCV`, `clone`, `cross_val_score` e DataFrame pandas con
  colonne categoriche, stringhe, booleane e valori mancanti.
- **Aperto.** Codice, pesi e l'intera pipeline di addestramento sotto Apache 2.0. Addestrato da zero: niente
  distillazione, niente pesi o output di altri modelli fondazionali tabellari.

## Benchmark

**TabArena-Lite, pipeline ufficiale, 38 dataset di classificazione, 99 metodi** (8 ottobre 2026).
LightPFN usa la configurazione di default con quattro estimatori, bagging a otto fold e il protocollo di
validazione ufficiale. Tutti i 38 task sono riusciti, nessuno è imputato. **25° su 99, Elo 1420 (+67 / -66)**.

| Metodo | Elo | IC 95% |
|---|---:|---:|
| TabICLv2 (default) | 1558 | +77 / -68 |
| TabDPT-1.3 (default) | 1467 | +78 / -54 |
| RealMLP (tuned + ensembled) | 1459 | +50 / -47 |
| **LightPFN (default)** | **1420** | **+67 / -66** |
| CatBoost (tuned) | 1378 | +58 / -55 |
| CatBoost (tuned + ensembled) | 1370 | +58 / -48 |
| LightGBM (tuned + ensembled) | 1365 | +52 / -42 |
| XGBoost (tuned + ensembled) | 1346 | +58 / -62 |
| CatBoost (default) | 1339 | +49 / -52 |
| XGBoost (default) | 1191 | +57 / -69 |
| LightGBM (default) | 1144 | +59 / -63 |
| RandomForest (default) | 1000 | +71 / -84 |

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_lite_pareto.png" width="100%" alt="TabArena-Lite: Elo rispetto al tempo di fit, LightPFN evidenziato tra i modelli fondazionali" /></p>

L'Elo stimato è sopra tutti i GBDT della classifica, anche ottimizzati e in ensemble; gli intervalli con
CatBoost tuned si sovrappongono, quindi il risultato non dimostra una vittoria netta. TabICLv2 e i modelli
fondazionali maggiori sono avanti. Il fit mediano è 0,65 s per 1.000 righe sulla RX 7900 XT; tra i metodi con
Elo almeno pari, solo TabDPT-1.3 ha un tempo riportato minore. I tempi della classifica provengono da hardware
diverso. È una valutazione Lite degli autori; i maintainer TabArena rieseguono il benchmark completo prima
dell'inserimento in classifica. [Protocollo, risultati e artefatti](https://github.com/GioOtto/LightPFN/blob/main/docs/it/RISULTATI.md).

## Installazione

```bash
pip install LightPFN             # CPU, CUDA o ROCm, tramite il PyTorch installato
pip install "LightPFN[vulkan]"   # aggiunge il backend GPU Vulkan (wgpu)
```

Serve Python 3.10 o successivo. Il primo `fit` scarica i pesi (18 MB) da
[Hugging Face](https://huggingface.co/ueuegio/LightPFN) a un commit fissato e li tiene in cache. Su una
macchina senza GPU conviene installare prima PyTorch dall'indice CPU, così non scarica le librerie CUDA:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Primi passi

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = LightPFNClassifier(n_estimators=4, random_state=0)
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

Un DataFrame pandas con categoriche e valori mancanti si passa così com'è:

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

Le colonne di tipo category, stringa, object o bool diventano codici ordinali delle categorie viste in
`fit`. Un dtype category conserva i livelli dichiarati, anche quelli inutilizzati. I valori mancanti e quelli
fuori da quel vocabolario diventano NaN, che il modello gestisce nativamente. Altri esempi in [examples/](https://github.com/GioOtto/LightPFN/tree/main/examples/) e nella [guida](https://github.com/GioOtto/LightPFN/blob/main/docs/it/GUIDA.md).

## Dispositivi

| `device=` | Dove gira |
|---|---|
| `"auto"` (predefinito) | una GPU CUDA o ROCm tramite PyTorch, altrimenti una GPU Vulkan, altrimenti la CPU |
| `"cpu"` | la CPU, tramite PyTorch |
| `"cuda"`, `"cuda:1"` | una GPU NVIDIA (CUDA) o AMD (ROCm) tramite PyTorch |
| `"vulkan"`, `"vulkan:1"` | qualsiasi GPU con driver Vulkan, tramite `lightpfn.vulkan` (serve `lightpfn[vulkan]`) |

La variabile d'ambiente `LIGHTPFN_DEVICE` sostituisce `"auto"`. Ogni dispositivo restituisce le stesse
probabilità a meno degli arrotondamenti in virgola mobile (entro 3e-5). Vedi il
[backend Vulkan](https://github.com/GioOtto/LightPFN/blob/main/docs/it/VULKAN.md).

## Quando usarlo

LightPFN è una buona scelta predefinita per tabelle di classificazione da 2 a 10 classi e fino a decine di
migliaia di righe, quando serve un modello forte senza tuning. Ha questi limiti:

- **Solo classificazione**, da 2 a 10 classi. La regressione arriva con la versione 2.
- **Tabelle grandi.** Oltre `max_context` righe (20.000 di default) ogni estimatore legge un sottoinsieme
  stratificato. Nel report, con tutto il training set come contesto, la sua AUC media è alla pari con
  CatBoost di default fino a 100.000 righe. Contesti più grandi costano di più; dimensioni oltre 100.000
  righe non sono state valutate.
- **Le colonne categoriche** vengono lette come codici ordinali. Sulle tabelle dominate da categoriche ad
  alta cardinalità CatBoost è avanti (fino a 4,5 punti di AUC su due task di TabArena).
- **Il tempo su CPU cresce con il contesto.** Su TabArena, quattro estimatori impiegano una mediana di 1 s per
  split sulle tabelle piccole, 15 s sulle medie e 54 s sulle grandi, contro 3, 4 e 7 s di CatBoost. Se il
  tempo conta, usa una GPU o `n_estimators=1` (circa quattro volte più veloce, poco meno accurato).

## Risultati

Baseline con le impostazioni di default. Differenze in punti di AUC (100 volte la differenza di AUC) con
intervalli bootstrap appaiati al 95%. Tabelle complete, risultati per task e tempi:
[docs/it/RISULTATI.md](https://github.com/GioOtto/LightPFN/blob/main/docs/it/RISULTATI.md).

**55 dataset OpenML-CC18 esterni a TabArena** (al massimo 1.000 righe e 100 feature, cross-validation a
cinque fold, un estimatore):

| Modello | AUC media | LightPFN meno modello |
|---|---:|---:|
| **LightPFN** | **0,911** | |
| CatBoost | 0,902 | +0,86 [0,42, 1,40] |
| Random forest | 0,893 | +1,76 |
| LightGBM | 0,891 | +2,02 |
| XGBoost (54 dataset: il suo wrapper fallisce su uno) | 0,889 | +2,06 [1,32, 2,94] |

**TabArena, 38 task di classificazione** (split ufficiali, prima ripetizione, eseguiti con il nostro harness,
che non è il protocollo della classifica ufficiale):

| Modello | AUC media | Rank medio su 7 | Errore minore di CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parametri, GPU) | 0,864 | 1,58 | 87% |
| **LightPFN, 4 estimatori** | **0,858** | **2,50** | **76%** |
| LightPFN, 1 estimatore | 0,857 | 3,37 | 68% |
| CatBoost | 0,855 | 3,50 | |
| LightGBM | 0,844 | 5,34 | 5% |
| Random forest | 0,838 | 5,89 | 3% |
| XGBoost | 0,833 | 5,82 | 11% |

## Come funziona

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/architecture.png" width="100%" alt="Architettura di LightPFN: embedding delle celle, due stadi per colonna, raffinamento e compressione delle righe, in-context learning, decoder a recupero" /></p>

Ogni cella viene codificata a partire dal suo valore, dal suo rango nella colonna e da un indicatore di
valore mancante. Due stadi per colonna con attenzione indotta leggono il contesto etichettato di ogni
colonna. Uno stadio per riga con quattro token di sintesi mescola le feature di ogni riga e la comprime in
un vettore. Sette blocchi di in-context learning fanno sì che ogni riga di test guardi le righe di training,
e un decoder a recupero trasforma l'attenzione in un voto sulle etichette di training.

Il preaddestramento ha usato 7,68 milioni di estrazioni da 4,03 milioni di problemi sintetici distinti: il 90% da un prior a grafi causali
strutturali e il 10% da un prior a regole (XOR, parità, tabelle di lookup, alberi) che insegna le interazioni
tra feature. Un secondo stadio ha addestrato su tabelle fino a 60.000 righe. Il
[report tecnico](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf) (in inglese) descrive modello, prior e protocollo di selezione;
[docs/it/ADDESTRAMENTO.md](https://github.com/GioOtto/LightPFN/blob/main/docs/it/ADDESTRAMENTO.md) spiega come riprodurre l'addestramento.

## Roadmap

La versione 2 aggiungerà:

- **Feature categoriche native.** Il modello vedrà la maschera delle categoriche fin dall'inizio del
  preaddestramento, e il prior conterrà colonne categoriche ad alta cardinalità con molti livelli rari,
  ciascuno con un piccolo effetto proprio. Un adapter addestrato sopra i pesi v1 congelati non ha aiutato,
  perché il prior v1 non ha colonne di quel tipo da cui imparare (report, Sezione 12).
- **Regressione.**

## Struttura della repository

| Percorso | Contenuto |
|---|---|
| `lightpfn/` | il pacchetto: modello, wrapper scikit-learn, dispositivi, backend Vulkan (`vulkan/`), e il codice di addestramento: prior (`prior/`), trainer (`train.py`), harness di valutazione (`eval/`) |
| `tests/` | test unitari e di equivalenza (CPU; i test Vulkan girano anche su un driver CPU) |
| `examples/` | esempi eseguibili |
| `docs/en/`, `docs/it/` | guida, risultati, backend Vulkan e addestramento, in inglese e in italiano |
| `paper/` | report tecnico: PDF, sorgente LaTeX, figure e dati dei grafici |
| `licenses/` | licenze delle dipendenze |

Il wheel pubblicato contiene solo il codice di inferenza; per l'addestramento serve il sorgente
(`pip install -e ".[train,eval]"`).

## Contribuire e supporto

Segnalazioni di bug e pull request circoscritte sono benvenute: vedi [CONTRIBUTING](https://github.com/GioOtto/LightPFN/blob/main/.github/CONTRIBUTING.md).
Le vulnerabilità passano da [SECURITY](https://github.com/GioOtto/LightPFN/blob/main/.github/SECURITY.md). Le modifiche tra versioni sono in
[CHANGELOG.md](https://github.com/GioOtto/LightPFN/blob/main/CHANGELOG.md).

## Citazione

```bibtex
@techreport{ottoboni2026lightpfn,
  title  = {A Sling Against Giants: {LightPFN}, a 4.6M-parameter tabular in-context classifier designed to stay small},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN}
}
```

Il pulsante "Cite this repository" di GitHub legge [CITATION.cff](https://github.com/GioOtto/LightPFN/blob/main/CITATION.cff).

## Licenza

Codice e pesi: [Apache License 2.0](https://github.com/GioOtto/LightPFN/blob/main/LICENSE), con l'avviso di attribuzione in [NOTICE](https://github.com/GioOtto/LightPFN/blob/main/NOTICE). Le
dipendenze mantengono le proprie licenze, elencate in [licenses/](https://github.com/GioOtto/LightPFN/tree/main/licenses/).
