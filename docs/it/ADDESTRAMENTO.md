# Addestramento e valutazione

[English](../en/TRAINING.md) · **Italiano**

Questa pagina spiega come sono stati addestrati i modelli rilasciati 1.1 e 1.5 e come riprodurli o addestrarne
una variante. Il [report tecnico di LightPFN 1.5](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
e il [report della 1.1](../../paper/LightPFN_report.pdf) (in inglese) spiegano le ragioni di ogni scelta.

- [Requisiti](#requisiti)
- [Prior sintetici](#prior-sintetici)
- [Generare un pool](#generare-un-pool)
- [Addestrare un modello](#addestrare-un-modello)
- [La ricetta rilasciata della 1.1](#la-ricetta-rilasciata-della-11)
- [La ricetta rilasciata della 1.5](#la-ricetta-rilasciata-della-15)
- [Valutazione](#valutazione)
- [Esportare i pesi per il pacchetto](#esportare-i-pesi-per-il-pacchetto)

## Requisiti

- Linux (generatore e flusso di dati usano i lock su file `fcntl`).
- Una GPU CUDA o ROCm con almeno 16 GB per l'addestramento; il run rilasciato della 1.1 ha usato due RTX 5090 e
  quello della 1.5 quattro RTX 5090.
- Una copia di questa repository: il wheel su PyPI contiene solo il codice di inferenza.

```bash
git clone https://github.com/GioOtto/LightPFN.git && cd LightPFN
pip install -e ".[train,eval,dev]"
```

L'extra `train` installa TabICL 2.2.0, il cui codice BSD-3-Clause dei modelli causali strutturali è riusato dal
prior a grafi (mai i suoi pesi o le sue predizioni). Pool, checkpoint e dati di valutazione finiscono in `data/`
e `runs/`, che non sono tracciate.

## Prior sintetici

| Prior | Versione | Contenuto |
|---|---|---|
| `graph` | 3 | modelli causali strutturali con grafi e funzioni dei nodi casuali (MLP, alberi, discretizzazioni, processi gaussiani), come in TabICLv2, con un filtro ExtraTrees che scarta i problemi non apprendibili. 90% del mix di classificazione rilasciato nella 1.1, e grafo di base del prior di regressione `r2` della 1.5 |
| `graph` | 4 | graph v3 con code pesanti, colonne categoriche ad alta cardinalità ottenute per quantizzazione, più problemi binari e target multiclasse divisi per quantili. Usato per la valutazione (D1) e per gli esperimenti sulle categoriche nella 1.1, e per il 90% del mix di preaddestramento di classificazione della 1.5 (con `p_binary=0.55`) |
| `rule` | 1 | feature di graph v4 etichettate da un oracolo su una-quattro colonne guida: XOR, parità, alberi poco profondi, tabelle di lookup, prodotti, AND, OR. Le colonne guida di XOR e parità sono bilanciate in modo che nessuna feature da sola predica l'etichetta. 10% del mix di classificazione nella 1.1 e nella 1.5 |
| `tree` | 4 | etichette da ensemble di alberi addestrati su feature di graph. Usato per la valutazione (D1) |
| `graph` (`r2`) | 3 | prior di regressione su graph v3 con `--reg-factors filter,noise,tails,discrete`: corregge filtro di apprendibilità, rumore osservativo, code pesanti e target discreti. Usato per addestrare il regressore 1.5 |

Le forme dei problemi vengono da preset: `s1b` da 256 a 2.048 righe in gruppi di 8 (preaddestramento), `s2` da
1.024 a 10.240 righe in gruppi di 4, `s4` da 4.096 a 60.000 righe in gruppi di 2 (stadio a contesto lungo).
`--geometry ratio` lega il numero di feature alle righe; `--geometry cap` estrae da 2 a 100 feature
indipendentemente dalle righe. Ogni problema ha un seme derivato dalle sue coordinate, così un pool è identico
byte per byte con qualsiasi numero di worker, e una generazione interrotta riprende da dove si era fermata.

## Generare un pool

```bash
python -m lightpfn.prior.generate --preset s1b --prior graph --prior-version 3 --geometry ratio \
    --p-binary 0.55 --max-features 100 --seed 0 --n-tasks 256000 --n-jobs 16 --out data/prior/s1b_graph
python -m lightpfn.prior.generate --preset s1b --prior rule --prior-version 1 --geometry ratio \
    --p-binary 0.78 --max-features 100 --seed 20261011 --n-tasks 40000 --n-jobs 16 --out data/prior/s1b_rule_v1
```

Sono i pool delle ablation del report della 1.1. Tempi di generazione e spazio dipendono dalla geometria e dalla
macchina; verifica ogni modifica ai prior ed esegui un piccolo pilot prima di generare un pool grande.

## Addestrare un modello

L'architettura del modello rilasciato nella 1.1 (B4 nel report della 1.1, 4.603.088 parametri) è la `Config` di
default più quattro impostazioni; il classificatore 1.5 (4.609.472 parametri) aggiunge
`"cat_adapter":true`:

```bash
B4='{"row_refine":true,"row_refine_rounds":3,"row_mode":"summary","icl_drop_blocks":1}'
python -m lightpfn.train --run my_run --pools data/prior/s1b_graph:0.9 data/prior/s1b_rule_v1:0.1 \
    --steps 8000 --warmup 800 --max-cells 400000 --config "$B4" --no-lite
```

È l'ablation da 8.000 passi del report della 1.1 (C2). Il trainer usa Muon per le matrici dei blocchi e AdamW
per il resto, una schedule a coseno, autocast bfloat16 e una media mobile (EMA) dei pesi, e scrive
`runs/train/my_run/ema_stepXXXXXX.pt` (pesi EMA) e `ckpt.pt` (tutto quello che serve per riprendere). Rilanciare
lo stesso comando riprende da `ckpt.pt`. Con più GPU si lancia tramite
`python -m torch.distributed.run --nproc_per_node=N -m lightpfn.train ...`; ogni passo contiene allora N volte i
problemi.

## La ricetta rilasciata della 1.1

<a id="la-ricetta-rilasciata"></a>

**Preaddestramento** (`final` nel report della 1.1): 120.000 passi da 64 problemi su due GPU, warmup 2.000,
graph v3 al 90% e rule v1 al 10%, con problemi nuovi generati in flusso da worker CPU. Il flusso
(`lightpfn.prior.stream`) genera blocchi da 7.500 passi in anticipo sul trainer e cancella ogni blocco dopo che
è stato letto due volte:

```bash
S=data/stream/final
python -m lightpfn.prior.stream init $PWD/$S --chunk-steps 7500 --chunks 16 --tasks-per-step 64 --passes 2 \
    --consumers final --prior v3=graph:3:0.55:0.9:3000000 --prior rule=rule:1:0.78:0.1:5000000
python -m lightpfn.prior.stream gen $PWD/$S --n-jobs 72 --ahead 3 &
python -m lightpfn.prior.stream train $PWD/$S --run final --gpu 0,1 --nproc 2 -- \
    --warmup 2000 --save-every 1000 --workers 4 --seed 0 --max-cells 800000 --config "$B4"
```

Ha richiesto circa 18 ore su due RTX 5090, con 72 processi CPU che generavano circa 120 problemi al secondo.

**Stadio a contesto lungo** (`final_long`, i pesi rilasciati della 1.1): altri 39.250 passi dal checkpoint
finale, su tabelle fino a 60.000 righe, learning rate 1e-4 con warmup 200 e coseno fino a 1e-5. Pool (il run ha
usato quanto i generatori avevano prodotto all'inizio di ciascuna parte dello stadio; i numeri sotto sono limiti
superiori):

```bash
G="python -m lightpfn.prior.generate --geometry cap --max-features 100"
$G --preset s2 --prior graph --prior-version 3 --p-binary 0.55 --seed 3000900 --n-tasks 345600 --out data/prior/s2cap_graph_v3
$G --preset s2 --prior rule --prior-version 1 --p-binary 0.78 --seed 5000900 --n-tasks 38400 --out data/prior/s2cap_rule_v1
$G --preset s4 --max-cells 800000 --prior graph --prior-version 3 --p-binary 0.55 --seed 3000902 --n-tasks 500000 --out data/prior/s4cap_graph_v3
$G --preset s4 --max-cells 800000 --prior rule --prior-version 1 --p-binary 0.78 --seed 5000902 --n-tasks 56000 --out data/prior/s4cap_rule_v1
```

L'addestramento parte dal checkpoint finale (pesi, EMA e stato dell'ottimizzatore) con il contatore dei passi
azzerato:

```bash
mkdir -p runs/train/final_long
python -c "import torch; c = torch.load('runs/train/final/ckpt.pt', weights_only=False); c['step'] = 0; torch.save(c, 'runs/train/final_long/ckpt.pt')"

python -m torch.distributed.run --standalone --nproc_per_node=2 -m lightpfn.train --run final_long \
    --pools data/prior/s1b_graph:0.135 data/prior/s1b_rule_v1:0.015 data/prior/s2cap_graph_v3:0.315 \
            data/prior/s2cap_rule_v1:0.035 data/prior/s4cap_graph_v3:0.45 data/prior/s4cap_rule_v1:0.05 \
    --sampling epoch --steps 39250 --warmup 200 --lr 1e-4 --groups-per-step 8 \
    --max-cells 1100000 --row-cost 8 --fit-rows --save-every 250 --workers 4 --seed 1 --config "$B4"
```

`--row-cost 8` conta ogni riga come otto celle in più quando si formano i micro-batch (un modello di memoria
misurato) e `--fit-rows` sottocampiona le righe di un problema che supererebbe il budget. Lo stadio della 1.1
non è stato eseguito in un pezzo solo: un limite della griglia CUDA (poi corretto) ha escluso le tabelle lunghe
per i primi 7.000 passi, e lo stadio è stato esteso due volte. Il report della 1.1 (Sezione 9) descrive la
sequenza esatta.

## La ricetta rilasciata della 1.5

La versione 1.5 usa il tronco B4 della 1.1 con l'adattatore categorico (`Config.cat_adapter=True`).
Classificatore (4.609.472 parametri) e regressore (4.772.519 parametri) hanno pesi separati, ciascuno
sotto 5 milioni di parametri. Non vengono usati né distillazione né pesi o output di altri modelli fondazionali
tabellari.

| Stadio | Ricetta |
|---|---|
| Classificazione principale | 120.000 passi x 64 problemi globali (7,68M estrazioni); `graph` v4 90%, `rule` v1 10%, entrambi con `p_binary=0.55`; flusso di 16 blocchi x 7.500 passi, due letture; `--cat-mask-drop 0.25` |
| Ottimizzazione principale | Muon + AdamW, autocast bfloat16, EMA, warmup 2.000, LR di picco 1e-3 con coseno fino a 1e-4, `--max-cells 800000` |
| Classificazione a contesto lungo | 15.000 passi dal checkpoint principale, tabelle fino a 60.000 righe; gruppi 15% `s1b` / 35% `s2cap` / 50% `s4cap`; warmup 200, LR 1e-4 con coseno fino a 1e-5, `--max-cells 1380000 --row-cost 10 --fit-rows` |
| Classificazione su tabelle larghe | 8.500 passi dal checkpoint a contesto lungo, stesse impostazioni dell'ottimizzatore (warmup 200, LR 1e-4 con coseno fino a 1e-5); gruppi 8% `s1b` / 17% `s2cap` / 25% `s4cap` / 50% `s2w`, dove `s2w` ha da 1.024 a 10.240 righe e da 64 a 500 feature (`--min-features 64 --max-features 500 --max-cells 1000000`) |
| Regressione | Fine-tuning del tronco di classificazione sul prior di regressione `r2` (`graph` v3 con `--reg-factors filter,noise,tails,discrete`, 2,5M problemi in flusso), decoder a recupero dei momenti e testa `quantiles` a 999 livelli (LR 1e-3, warmup 200); checkpoint separato |
| Regressione a contesto lungo | 5.000 passi con stadio di contesto fino a 60.000 righe |

- **Quota di problemi binari uguale (`p_binary=0.55`):** nella 1.1 `rule` v1 usava `p_binary=0.78`. Confronti
  controllati a 2.000 passi hanno mostrato che alzare `p_binary` da `0.55` a `0.78` penalizza i problemi
  multiclasse, mentre `graph` v4 a `p_binary=0.55` pareggia `graph` v3 su D1 e D3 e migliora i problemi con
  categoriche ad alta cardinalità. La 1.5 imposta quindi `p_binary=0.55` sia in `graph` v4 sia in `rule` v1.
- **Drop della maschera categorica (`--cat-mask-drop 0.25`):** durante il preaddestramento la maschera delle
  categoriche viene omessa nel 25% dei problemi, così il modello gestisce bene anche colonne categoriche
  codificate come interi senza maschera, mentre `cat="auto"` fornisce la maschera in inferenza.
- **Prior di regressione (`r2`) e testa:** l'analisi del prior di regressione `graph` v3 di base rispetto ai
  benchmark reali mostrava il 36% di problemi non apprendibili (contro il 4% nei dati reali) e troppo pochi
  target a code pesanti o discreti. Il prior `r2` corregge filtro di apprendibilità, rumore osservativo, code
  pesanti e target discreti. Il decoder di regressione legge i momenti del target dal contesto con una sola
  attenzione fusa e la testa `quantiles` restituisce 999 quantili ordinati da `.001` a `.999`. L'affinamento del
  tronco di classificazione preaddestrato raggiunge un CRPS migliore dell'addestramento da zero a parità di
  calcolo. La regola `log(y)` / `-y` del secondo membro (`log_skew=2.0`) è una ricetta di sola inferenza e non
  modifica il training. Entrambi i checkpoint rilasciati usano i pesi EMA.

## Valutazione

| Insieme | Contenuto | Comandi |
|---|---|---|
| D1 | 1.536 problemi sintetici tenuti da parte: famiglie graph v3, graph v4, tree e rule, con un seme che nessun pool di training usa | `python -m lightpfn.eval.dev build`, poi `python -m lightpfn.eval.dev evaluate --checkpoint CKPT --run NOME` |
| D2 | 25 configurazioni di sonde (XOR, parità, tabelle di lookup, colonne esca...) con cinque semi | eseguito da `lightpfn.eval.dev evaluate` |
| D3 | 55 dataset di classificazione OpenML-CC18 non in TabArena, al massimo 1.000 righe e 100 feature, CV a cinque fold | `python -m lightpfn.eval.external build`, `run --checkpoint CKPT --name NOME`, `report` |
| scala | i 15 dataset di D3 con almeno 5.000 righe, training set da 1.000 a 32.000 righe e tutte le righe | `python -m lightpfn.eval.scale run --checkpoint CKPT --name NOME`, poi `report --models NOME catboost` |
| D4 / Dcat | 13 dataset OpenML da 50.000 a 2,2 milioni di righe esterni a TabArena e D3 (Dcat: i 6 task con colonne categoriche ad alta cardinalità) | `python -m lightpfn.eval.large build`, `run --checkpoint CKPT --name NOME`, `report --models NOME catboost` |
| D3-reg / D3-reg-L | 23 task di regressione OpenML-CTR23 esterni a TabArena (fino a 800 righe di contesto, CV a cinque fold) e 15 task CTR23 più grandi (da 4.000 a 40.000 righe di contesto, due split) | `python -m lightpfn.eval.regression` |
| TabArena | i 38 task di classificazione e i 13 di regressione di TabArena v0.1, protocolli lite e full (il nostro harness, non la pipeline ufficiale) | `python -m lightpfn.eval.data`, poi `python -m lightpfn.eval.harness --checkpoint CKPT --name NOME --n-estimators 4 --mode full` |

La selezione dei modelli in entrambe le serie ha usato i set di sviluppo (D1, D2, D3, Dcat, D3-reg, D3-reg-L)
con confronti bootstrap appaiati. TabArena viene consultato solo per i finalisti; i risultati pubblicati come
locali non equivalgono alla classifica ufficiale di TabArena. Una valutazione esplorativa precedente di r2 nella
1.1 è dichiarata nel report della 1.1, Sezione 7. Le baseline (CatBoost, LightGBM, XGBoost, random forest)
passano dallo stesso harness con le impostazioni di default:
`python -m lightpfn.eval.external run --models catboost lightgbm xgboost rf`.

Usa un `--name` diverso per ogni checkpoint e configurazione degli estimatori. L'harness riprende gli split
riusciti dal CSV identificato da `--name`; non controlla se il checkpoint o la configurazione sono cambiati.

## Esportare i pesi per il pacchetto

```python
from lightpfn import load_model, save_model

# sottocartelle di classificazione e regressione della 1.5
cls_model = load_model("runs/train/cls_wide/ema_step008500.pt")  # pesi EMA, caricamento weights_only
reg_model = load_model("runs/train/reg_long/ema_step005000.pt")
save_model(cls_model, "weights/classifier")                      # model.safetensors + config.json
save_model(reg_model, "weights/regressor")

# checkpoint della 1.1
v1_model = load_model("runs/train/final_long/ema_step039250.pt")
save_model(v1_model, "my_weights/")
```

Poi `LightPFNClassifier(checkpoint="weights/classifier")` e `LightPFNRegressor(checkpoint="weights/regressor")`
(oppure `LightPFNClassifier(checkpoint="my_weights/")`) li caricano direttamente.
