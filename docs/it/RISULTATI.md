# Risultati

[English](../en/RESULTS.md) · **Italiano**

Questa pagina riporta i risultati delle valutazioni locali di LightPFN 1.5 e i risultati completi di benchmark
di LightPFN 1.1. Protocolli e ablation sono descritti nel
[report tecnico di LightPFN 1.5](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf) e
nel [report della 1.1](../../paper/LightPFN_report.pdf) (in inglese). La voce nella classifica ufficiale di
TabArena è LightPFN 1.1 (il checkpoint a contesto lungo LC39k nel report della 1.1, 4.603.088 parametri);
LightPFN 1.5 non ha ancora una voce nella classifica ufficiale, e le sue misure riportate qui sotto sono
esecuzioni locali.

- [Risultati della 1.5 (valutazioni locali)](#risultati-della-15-valutazioni-locali)
  - [Regola finale del secondo membro: verifica CPU fp32](#regola-finale-del-secondo-membro-verifica-cpu-fp32)
- [Risultati della 1.1](#risultati-della-11)
  - [Insiemi di valutazione](#insiemi-di-valutazione)
  - [Tabelle piccole: 55 dataset OpenML esterni a TabArena](#tabelle-piccole-55-dataset-openml-esterni-a-tabarena)
  - [TabArena, classifica ufficiale](#tabarena-classifica-ufficiale)
  - [TabArena, 38 task di classificazione con il nostro harness](#tabarena-38-task-di-classificazione-con-il-nostro-harness)
  - [Tabelle più grandi](#tabelle-più-grandi)
  - [Costo](#costo)
  - [Feature categoriche: cosa non ha funzionato](#feature-categoriche-cosa-non-ha-funzionato)

## Risultati della 1.5 (valutazioni locali)

<a id="risultati-della-20"></a>

Risultati locali raccolti al 9 novembre 2026, non classifica ufficiale. Le sezioni successive sono storiche e
riguardano LightPFN 1.1.

| Misura (1.5, locale) | Risultato | Protocollo e limite |
|---|---|---|
| Lite classificazione, classificatore finale | Elo 1436 (+74 / -65), 1.1 1423; classificatore 1.5 precedente 1416 sulla stessa GPU (1415 su RX 7900 XT); 24 task con errore minore, 14 maggiore | 38 task, primo split, RTX 5090; snapshot delle baseline pubblicate |
| Lite regressione, esecuzione finale | Elo 1450 (+107 / -103), CatBoost default 1416 | 13 task, fold 0, RX 7900 XT; regola finale con soglia di asimmetria |
| Due membri rispetto a uno, studio precedente | RMSE -1,03% [-1,73, -0,40] D3-reg; -6,72% [-14,12, -1,31] D3-reg-L | 23 e 15 task di sviluppo, contesto completo; regola precedente |
| Classificazione, velocità train / predict | 1,89x / 2,04x rispetto alla 1.1.0 | Medie geometriche, 8 task, 8 fold x 4 membri, stessa RTX 5090, soglia fp16 misurata di 8.192 celle |
| Fine-tuning sulle tabelle larghe, finale meno classificatore 1.5 precedente | D3 > 100 feature: AUC +1,93 punti [-0,06, +5,82], log loss -0,048 [-0,108, -0,004]; D3 +0,01 [-0,06, +0,09]; Dcat 10k +0,07 [+0,00, +0,16]; D4 +0,12 [+0,05, +0,21]; errore relativo sul primo split di TabArena -0,85% [-1,83, -0,04] | Quattro membri, bootstrap appaiato al 95%; 9, 55, 6 e 13 dataset, 38 task TabArena (osservati durante il fine-tuning, non indipendenti) |
| Classificazione, classificatore finale, un membro | D3 AUC 0,9102, Dcat 10k 0,7082 | Stessi protocolli delle righe sotto |
| Classificazione, classificatore 1.5 prima del fine-tuning sulle tabelle larghe | D3 AUC 0,9103 (1.1.0: 0,9109; CatBoost default: 0,9023) | 55 dataset OpenML-CC18 non presenti in TabArena, un membro, locale |
| Task categorici ad alta cardinalità (Dcat) | AUC 0,7075 a 10k righe, 0,7134 a 25k (1.1.0: 0,7047 / 0,7098; CatBoost default: 0,7031 / 0,7118) | 6 dataset con colonne categoriche ad alta cardinalità, un membro, locale |
| Tabelle grandi (D4) | AUC 0,8200 (1.1.0: 0,8188; CatBoost default: 0,8189) | 46 celle, un membro, locale |

Scalando i tempi ufficiali della 1.1.0 con questi rapporti di velocità si ottiene circa 0,14 s in train e circa
0,014 s in predict per 1.000 righe: è un'estrapolazione tra hardware diversi, non una misura ufficiale della
1.5. La soglia fp16 del pacchetto è ora 2.048 celle di contesto per membro. Il default finale di
`LightPFNRegressor` applica `log(y)` solo con target positivi e asimmetria campionaria corretta almeno 2
(`log_skew=2.0`), altrimenti `-y` (`"flip"`). Questa regola è stata scelta sui set di sviluppo ed è usata
nella Lite finale riportata sopra. I precedenti wrapper avevano Elo di regressione 1315 e 1351 contro uno
snapshot più piccolo delle baseline: le variazioni di Elo non sono direttamente confrontabili. Le posizioni
locali finali sono 34/115 in classificazione (classificatore finale; 35/115 quello precedente) e 54/111 in regressione, incluse le baseline imputate. Nessun
risultato della 1.5 è imputato. I tempi locali sono 0,31/0,041 s per 1.000 righe in train/predict per la
classificazione (classificatore precedente, RX 7900 XT; classificatore finale su RTX 5090: 0,29/0,023) e 0,21/0,033 per la regressione; erano attivi altri job CPU, e le baseline pubblicate usano
hardware diverso.

La regola finale raggiunge CRPS/RMSE normalizzati medi di **0,1748/0,4042** su D3-reg e **0,1546/0,3691**
su D3-reg-L. D3-reg ha al massimo 800 righe di contesto; D3-reg-L ha due split per ciascuna di 15 sorgenti
con almeno 4.000 righe finite, fino a 40.000 righe di contesto e 5.000 di test. Sono risultati a pesi fissi
su CPU fp32. CatBoost default ha RMSE 0,4794 e 0,3689 nei rispettivi protocolli; LightPFN vince 22/23 e
11/15 task, con rapporti geometrici medi RMSE di 0,791 e 0,921. Nella Lite finale il rapporto è 0,972
(8/13 vittorie), mentre Kumo-Tabular-Small e TabICLv2, più grandi, battono LightPFN su ogni task.
I quantili degli alberi su D3-reg usano la calibrazione dei residui su dati separati; D3-reg-L usa fit
puntuali e Lite il bagging TabArena. I protocolli e le aggregazioni dei benchmark sono diversi.

### Regola finale del secondo membro: verifica CPU fp32

Tutti i 23 task D3-reg (115 split) e i 15 D3-reg-L (30 split), quattro thread CPU. Variazione relativa di RMSE e
CRPS: prima media degli split per task, poi media dei rapporti tra task; IC 95% bootstrap appaiato per task
(10.000 campioni). Negativo è meglio. I CSV precedenti usano il percorso GPU; il riferimento CPU riesegue in
fp32 la vecchia regola basata sui target positivi per isolare il cambio di regola.

| Insieme | Riferimento | RMSE % [IC 95%] | CRPS % [IC 95%] |
|---|---|---:|---:|
| D3-reg (23 task) | CSV del pacchetto precedente | -0,74 [-1,61, -0,13] | -0,75 [-1,44, -0,19] |
| D3-reg (23 task) | Vecchia regola, CPU fp32 | -0,70 [-1,52, -0,10] | -0,72 [-1,40, -0,18] |
| D3-reg-L (15 task) | CSV del pacchetto precedente | -2,33 [-5,22, -0,39] | -2,03 [-4,54, -0,30] |
| D3-reg-L (15 task) | Vecchia regola, CPU fp32 | -1,78 [-4,05, -0,15] | -1,55 [-3,45, -0,11] |

La soglia di asimmetria è stata scelta su questi task di sviluppo, quindi le stime sono ottimistiche. Sono
esecuzioni locali di sviluppo, non risultati ufficiali di TabArena.
[Riepilogo numerico CSV](../assets/v2_regression_quality.csv).

## Risultati della 1.1

I protocolli sono descritti nel [report della 1.1](../../paper/LightPFN_report.pdf) (in inglese). Le tabelle qui
sotto riportano i risultati dei pesi **LightPFN 1.1** (il checkpoint a contesto lungo LC39k nel report della
1.1, 4.603.088 parametri) dal report e dai CSV collegati. La sezione ufficiale TabArena include baseline di
default, ottimizzate e in ensemble; gli altri confronti usano baseline di default. Le differenze di AUC sono in
punti (100 volte la differenza di AUC), con intervalli bootstrap appaiati al 95% su task o dataset.

## Insiemi di valutazione

Le scelte di progetto sono state prese su tre insiemi tenuti separati da TabArena: D1, 1.536 problemi sintetici
tenuti da parte; D2, sonde sui meccanismi (XOR, parità, tabelle di lookup tra colonne di rumore); D3, 55 dataset
di classificazione OpenML-CC18 che non sono in TabArena. Per la serie 1.1, dopo aver introdotto il protocollo di
selezione, TabArena è stato guardato solo per i due finalisti, dopo la scelta tra i due. Una valutazione
esplorativa precedente di r2 precede questo protocollo ed è dichiarata nel report tecnico della 1.1, Sezione 7.

## Tabelle piccole: 55 dataset OpenML esterni a TabArena

Risultati per **LightPFN 1.1** (al massimo 1.000 righe e 100 feature casuali per dataset, cross-validation
stratificata a cinque fold, un estimatore):

| Modello | AUC media | LightPFN 1.1 meno modello | LightPFN 1.1 più alto su |
|---|---:|---:|---:|
| **LightPFN 1.1** | **0,9109** | | |
| CatBoost | 0,9023 | +0,86 [0,42, 1,40] | 80% dei dataset |
| Random forest | 0,8933 | +1,76 [1,03, 2,62] | 91% |
| LightGBM | 0,8907 | +2,02 [1,14, 3,16] | 91% |
| XGBoost, 54 dataset | 0,8889 | +2,06 [1,32, 2,94] | 94% |

Il wrapper categorico di XGBoost fallisce su un dataset (*sick*), escluso da entrambi i lati del confronto. Il
vantaggio su CatBoost vale sia sui dataset binari (+0,80) sia sui multiclasse (+0,92), con intervalli sopra zero.

[CSV dei confronti D3](../assets/d3_baselines.csv), inclusi intervalli bootstrap e percentuali di vittorie.

## TabArena, classifica ufficiale

Classifica dell'11 ottobre 2026 ([Space di TabArena](https://huggingface.co/spaces/TabArena/leaderboard), commit
dei dati `0aac376`). I maintainer di TabArena hanno eseguito **LightPFN 1.1** sui 38 dataset di classificazione
con il protocollo ufficiale (bagging a otto fold, validazione `8x1`) nella configurazione di default: quattro
estimatori, un checkpoint, nessuno spazio di ricerca. Hanno marcato il risultato come verificato; nessuno dei
suoi risultati di classificazione è imputato. Le viste qui sotto sono Classification con i modelli imputati
esclusi. La vista di default della pagina comprende anche i 13 dataset di regressione, che LightPFN 1.1 non
supportava, quindi lì vengono imputati (42° su 99).

| Vista (LightPFN 1.1) | Dataset | Posizione | Elo (IC 95%) |
|---|---:|---:|---:|
| **Classificazione, tutti gli split** | 38 | **21° su 92** | **1441 (+69 / -79)** |
| Classificazione, Lite (primo split) | 38 | 20° su 92 | 1426 (+71 / -70) |
| Binaria, tutti gli split | 30 | 23° su 92 | 1419 (+77 / -67) |
| Multiclasse, tutti gli split | 8 | 17° su 92 | 1560 (+121 / -70) |
| Dataset piccoli, tutti gli split | 28 | 19° su 92 | 1457 (+79 / -64) |
| Dataset medi, tutti gli split | 10 | 44° su 93 | 1465 (+156 / -145) |

L'Elo è calcolato tra i metodi di ciascuna vista, quindi i valori non sono confrontabili tra righe.

![Elo con intervalli al 95% per una selezione di metodi, LightPFN 1.1 in blu](../assets/tabarena_elo.png)

Metodi selezionati, classificazione, tutti gli split. I tempi sono le mediane di TabArena in secondi per 1.000
righe; le versioni ottimizzate includono il tempo della ricerca.

| Posizione | Metodo | Elo | IC 95% | Training | Predizione |
|---:|---|---:|---:|---:|---:|
| 1 | LimiX-2 (default) | 1927 | +137 / -77 | 36,96 | 10,081 |
| 2 | Kumo-Tabular (default) | 1918 | +163 / -112 | 2,79 | 0,272 |
| 4 | TabPFN-3.5 (default) | 1845 | +115 / -70 | 1,85 | 0,518 |
| 14 | TabICLv2 (default) | 1575 | +72 / -69 | 0,93 | 0,142 |
| 18 | TabDPT-1.3 (default) | 1524 | +75 / -57 | 0,62 | 0,172 |
| 19 | RealMLP (tuned + ensembled) | 1474 | +53 / -43 | 2879,46 | 12,484 |
| 20 | TabM (tuned + ensembled) | 1452 | +56 / -40 | 2168,86 | 2,347 |
| **21** | **LightPFN 1.1 (default)** | **1441** | **+69 / -79** | **0,26** | **0,029** |
| 22 | LightGBM (tuned + ensembled) | 1426 | +43 / -29 | 382,09 | 1,487 |
| 24 | CatBoost (tuned + ensembled) | 1410 | +53 / -44 | 1107 | 0,329 |
| 27 | CatBoost (tuned) | 1400 | +47 / -43 | 1107 | 0,038 |
| 31 | CatBoost (default) | 1387 | +44 / -45 | 5,41 | 0,026 |
| 35 | XGBoost (tuned + ensembled) | 1373 | +40 / -46 | 685,87 | 1,454 |
| 60 | XGBoost (default) | 1230 | +39 / -46 | 1,77 | 0,117 |
| 66 | LightGBM (default) | 1191 | +50 / -41 | 1,79 | 0,12 |
| 87 | RandomForest (default) | 1000 | +62 / -65 | 0,38 | 0,037 |

L'Elo stimato di LightPFN 1.1 è sopra tutti i metodi ad alberi, anche ottimizzati e in ensemble. Il suo
intervallo si sovrappone a quelli di CatBoost di default e delle versioni ottimizzate dei gradient boosting,
quindi contro di loro non è una vittoria netta; XGBoost e LightGBM di default sono più di 200 punti sotto, con
intervalli separati. Nella matrice delle vittorie di TabArena LightPFN 1.1 batte CatBoost, LightGBM e XGBoost
ottimizzati e in ensemble nel 58%, 58% e 64% dei confronti, e TabICLv2 nel 25%. I primi 18 posti sono tutti
modelli fondazionali. Il punto debole sono i 10 dataset medi (44° su 93, dove CatBoost di default è 17°),
coerente con i risultati sulle tabelle più grandi più sotto.

![Elo rispetto al tempo mediano di addestramento per 92 metodi, LightPFN 1.1 sul fronte di Pareto](../assets/tabarena_pareto.png)

LightPFN 1.1 è sul fronte di Pareto tra Elo e tempo di addestramento e tra Elo e tempo di predizione: solo KNN
si addestra più in fretta e solo EBM e CatBoost di default predicono più in fretta, tutti con un Elo più basso.
TabArena ha misurato LightPFN 1.1 su GPU e i metodi ad alberi su CPU, quindi l'asse dei tempi confronta le
configurazioni come sono state eseguite, non gli algoritmi a parità di hardware.

[CSV della classificazione, tutti gli split](../assets/tabarena_classification_all_splits.csv),
[CSV della classificazione, Lite](../assets/tabarena_classification_lite.csv) (come pubblicati da TabArena).

### Valutazione Lite precedente degli autori (8 ottobre 2026)

Prima della run dei maintainer abbiamo eseguito noi il benchmark Lite su LightPFN 1.1, con
`TabArenaV0pt1ExperimentBundle(models=[("LightPFN", 0)])` e `subset=["lite", "classification"]`: stessa
configurazione, primo split ufficiale di ciascun dataset. La cache del contesto è disattivata nei fold e attiva
nel refit. Tutti i 38 task sono riusciti, nessuno è imputato. Rispetto alla classifica di quel giorno era
**25° su 99, Elo 1420 (+67 / -66)**. Hardware: Intel i7-13700KF, 32 GB RAM, AMD RX 7900 XT 20 GB, Linux,
ROCm 7.2, PyTorch 2.13.0, Python 3.12.15; fit mediano 0,65 s e predizione mediana 0,086 s per 1.000 righe.

[CSV della classifica di quel giorno](../assets/tabarena_lite_leaderboard.csv), [script e risultati completi (ZIP)](https://github.com/GioOtto/LightPFN/releases/download/v1.0.0/LightPFN-1.0.0-tabarena-lite.zip).
Lo script importa `torch` prima di AutoGluon per far rilevare la GPU AMD tramite ROCm.

## TabArena, 38 task di classificazione con il nostro harness

I 38 task di classificazione di TabArena v0.1, split ufficiali OpenML, prima ripetizione (114 split), valutati
su **LightPFN 1.1**. È il nostro harness, non il protocollo della classifica ufficiale (che fa bagging,
ottimizza i modelli e ordina per Elo); i 13 task di regressione non sono valutati qui. L'errore è 1 - AUC per i
task binari e la log loss per i multiclasse, come in TabArena; i rank sono calcolati sull'errore.

| Modello | AUC media | Rank medio su 7 | Errore minore di CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parametri, GPU) | 0,8637 | 1,58 | 87% |
| **LightPFN 1.1, 4 estimatori** | **0,8581** | **2,50** | **76%** |
| LightPFN 1.1, 1 estimatore | 0,8568 | 3,37 | 68% |
| CatBoost | 0,8553 | 3,50 | |
| LightGBM | 0,8440 | 5,34 | 5% |
| Random forest | 0,8383 | 5,89 | 3% |
| XGBoost | 0,8332 | 5,82 | 11% |

La differenza di AUC media da CatBoost è +0,28 punti [-0,26, 0,80]: LightPFN 1.1 vince più task, ma la
differenza media non è significativa. TabICLv2 è avanti di 0,56 punti [0,25, 0,98].

Per gruppo (LightPFN 1.1 con 4 estimatori contro CatBoost):

| Gruppo | Task | Rank medio, LightPFN 1.1 / CatBoost / TabICLv2 | Punti di AUC rispetto a CatBoost | Task vinti |
|---|---:|---|---:|---:|
| binari | 30 | 2,60 / 3,30 / 1,57 | +0,25 | 70% |
| multiclasse | 8 | 2,13 / 4,25 / 1,63 | +0,38 | 100% |
| meno di 2.500 righe | 12 | 2,25 / 4,25 / 1,42 | +1,18 | 100% |
| da 2.500 a 10.000 righe | 10 | 2,70 / 3,20 / 1,80 | +0,28 | 60% |
| 10.000 righe o più | 16 | 2,56 / 3,13 / 1,56 | -0,40 | 69% |
| classe minoritaria dall'1 al 10% | 11 | 2,55 / 2,91 / 1,82 | -0,35 | 64% |

Le perdite maggiori della 1.1 sono su tabelle soprattutto categoriche: *in_vehicle_coupon_recommendation*
(-4,49 punti), *Amazon_employee_access* (-3,95) e *Diabetes130US* (-2,35).

<details>
<summary>AUC media per task (in grassetto il migliore), ordinati per LightPFN 1.1 meno CatBoost</summary>

| Task | Righe di training | Feature | Classi | LightPFN 1.1 x4 | CatBoost | TabICLv2 | LightGBM | XGBoost | RF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jm1 | 7,257 | 21 | 2 | **0.7781** | 0.7421 | 0.7777 | 0.7385 | 0.7170 | 0.7505 |
| seismic-bumps | 1,723 | 15 | 2 | 0.7819 | 0.7501 | **0.7867** | 0.7267 | 0.7094 | 0.7301 |
| Marketing_Campaign | 1,494 | 25 | 2 | 0.9236 | 0.8936 | **0.9291** | 0.8912 | 0.7668 | 0.8878 |
| coil2000_insurance_policies | 6,548 | 85 | 2 | 0.7708 | 0.7444 | **0.7709** | 0.7299 | 0.7120 | 0.6913 |
| blood-transfusion-service-center | 499 | 4 | 2 | 0.7454 | 0.7199 | **0.7522** | 0.6774 | 0.6610 | 0.6767 |
| hazelnut-spread-contaminant-detection | 1,600 | 30 | 2 | 0.9882 | 0.9700 | **0.9942** | 0.9699 | 0.9720 | 0.9548 |
| MIC | 1,133 | 111 | 8 | 0.8669 | 0.8514 | **0.8768** | 0.8472 | 0.8476 | 0.8376 |
| Is-this-a-good-customer | 1,149 | 13 | 2 | 0.7465 | 0.7352 | **0.7497** | 0.6948 | 0.7036 | 0.7188 |
| maternal_health_risk | 676 | 6 | 3 | 0.9596 | 0.9494 | **0.9615** | 0.9463 | 0.9530 | 0.9483 |
| qsar-biodeg | 703 | 41 | 2 | **0.9435** | 0.9349 | 0.9430 | 0.9284 | 0.9279 | 0.9328 |
| Fitness_Club | 1,000 | 6 | 2 | 0.8220 | 0.8135 | **0.8220** | 0.7808 | 0.7710 | 0.7890 |
| students_dropout_and_academic_success | 2,950 | 36 | 3 | 0.8973 | 0.8891 | **0.9006** | 0.8873 | 0.8814 | 0.8810 |
| credit_card_clients_default | 20,000 | 23 | 2 | 0.7890 | 0.7819 | **0.7910** | 0.7792 | 0.7599 | 0.7621 |
| taiwanese_bankruptcy_prediction | 4,546 | 94 | 2 | **0.9441** | 0.9373 | 0.9434 | 0.9360 | 0.9302 | 0.9165 |
| heloc | 6,973 | 23 | 2 | 0.8015 | 0.7950 | **0.8022** | 0.7878 | 0.7712 | 0.7900 |
| Bank_Customer_Churn | 6,667 | 10 | 2 | 0.8714 | 0.8666 | **0.8732** | 0.8581 | 0.8386 | 0.8508 |
| credit-g | 667 | 20 | 2 | 0.7855 | 0.7809 | **0.7928** | 0.7734 | 0.7664 | 0.7651 |
| website_phishing | 902 | 9 | 3 | **0.9802** | 0.9757 | 0.9801 | 0.9756 | 0.9741 | 0.9717 |
| online_shoppers_intention | 8,220 | 17 | 2 | 0.9366 | 0.9324 | **0.9377** | 0.9302 | 0.9198 | 0.9237 |
| diabetes | 512 | 8 | 2 | 0.8415 | 0.8390 | **0.8426** | 0.8130 | 0.7995 | 0.8213 |
| GiveMeSomeCredit | 100,000 | 10 | 2 | 0.8656 | 0.8634 | **0.8669** | 0.8636 | 0.8542 | 0.8408 |
| anneal | 599 | 38 | 5 | **1.0000** | 0.9982 | 0.9994 | 0.9933 | 1.0000 | 0.9988 |
| HR_Analytics_Job_Change_of_Data_Scientists | 12,772 | 12 | 2 | 0.8050 | 0.8035 | **0.8056** | 0.7966 | 0.7709 | 0.7852 |
| bank-marketing | 30,141 | 13 | 2 | **0.7647** | 0.7632 | 0.7640 | 0.7559 | 0.7390 | 0.7247 |
| SDSS17 | 52,036 | 11 | 3 | 0.9963 | 0.9952 | **0.9964** | 0.9928 | 0.9924 | 0.9941 |
| E-CommereShippingData | 7,333 | 10 | 2 | **0.7472** | 0.7462 | 0.7444 | 0.7377 | 0.7415 | 0.7380 |
| APSFailure | 50,667 | 170 | 2 | 0.9924 | 0.9921 | **0.9946** | 0.9903 | 0.9871 | 0.9881 |
| splice | 2,127 | 60 | 3 | 0.9953 | 0.9953 | **0.9960** | 0.9945 | 0.9951 | 0.9920 |
| NATICUSdroid | 4,994 | 86 | 2 | 0.9853 | 0.9857 | **0.9880** | 0.9839 | 0.9845 | 0.9780 |
| churn | 3,334 | 19 | 2 | 0.9265 | 0.9270 | **0.9338** | 0.9160 | 0.9011 | 0.9190 |
| customer_satisfaction_in_airline | 86,587 | 21 | 2 | 0.9921 | 0.9945 | **0.9952** | 0.9927 | 0.9937 | 0.9923 |
| hiva_agnostic | 2,564 | 1,617 | 3 | 0.4694 | 0.4804 | 0.4808 | 0.4935 | 0.4952 | **0.5185** |
| Bioresponse | 2,501 | 1,776 | 2 | 0.8539 | 0.8700 | 0.8606 | **0.8719** | 0.8690 | 0.8677 |
| polish_companies_bankruptcy | 3,940 | 64 | 2 | 0.9429 | 0.9602 | **0.9834** | 0.9582 | 0.9558 | 0.9323 |
| kddcup09_appetency | 33,334 | 212 | 2 | 0.8218 | **0.8414** | 0.8237 | 0.7696 | 0.7505 | 0.7251 |
| Diabetes130US | 47,679 | 47 | 2 | 0.6451 | **0.6686** | 0.6641 | 0.6270 | 0.5637 | 0.6182 |
| Amazon_employee_access | 21,846 | 9 | 2 | 0.8487 | **0.8882** | 0.8533 | 0.8456 | 0.8587 | 0.8401 |
| in_vehicle_coupon_recommendation | 8,456 | 24 | 2 | 0.7812 | 0.8261 | **0.8428** | 0.8186 | 0.8272 | 0.8018 |

</details>

[CSV delle AUC per dataset del nostro harness](../assets/tabarena_harness.csv).

## Tabelle più grandi

Risultati per **LightPFN 1.1**:

**Curve di apprendimento.** I 15 dataset di D3 con almeno 5.000 righe, due ripetizioni, tutto il training set come
contesto (un estimatore). Differenza da CatBoost di default:

| Righe di training | Dataset | AUC CatBoost | LightPFN 1.1 meno CatBoost |
|---|---:|---:|---:|
| 1.000 | 15 | 0,8959 | +0,46 [0,13, 0,95] |
| 2.000 | 15 | 0,9071 | +0,51 [0,22, 0,84] |
| 4.000 | 15 | 0,9187 | +0,33 [-0,08, 0,74] |
| 8.000 | 9 | 0,9085 | +0,15 [-0,42, 0,74] |
| 16.000 | 6 | 0,8723 | -0,38 [-1,70, 0,92] |
| 32.000 | 6 | 0,8784 | -0,42 [-1,41, 0,70] |
| tutte (da 4.324 a 94.320) | 15 | 0,9313 | +0,31 [-0,23, 0,86] |

**Dataset grandi (D4).** 13 dataset OpenML da 50.000 a 2,2 milioni di righe esterni a TabArena e D3, 5.000 righe
di test, una ripetizione, tutto il training set come contesto:

| Righe di training | Dataset | AUC CatBoost | AUC LightPFN 1.1 | LightPFN 1.1 meno CatBoost | LightGBM / XGBoost meno CatBoost |
|---|---:|---:|---:|---:|---:|
| 10.000 | 13 | 0,8141 | 0,8168 | +0,28 [-0,53, 1,20] | -4,33 / -3,30 |
| 25.000 | 13 | 0,8225 | 0,8234 | +0,09 [-0,81, 1,08] | -3,52 / -2,72 |
| 50.000 | 12 | 0,8311 | 0,8291 | -0,20 [-1,17, 0,76] | -4,01 / -2,78 |
| 100.000 | 8 | 0,8027 | 0,7993 | -0,34 [-1,95, 1,29] | -0,83 / -2,51 |

LightPFN 1.1 è avanti su *porto-seguro*, *jannis*, *covertype* e *Higgs* e indietro su *albert*, *road-safety* e
*kick*, dove contano colonne categoriche ad alta cardinalità.

## Costo

**TabArena (1.1), tempo mediano di fit più predict per split** (protocollo full). Le macchine sono diverse, quindi
il confronto tra righe è solo indicativo: gli ensemble di alberi hanno girato su un Intel i7-13700KF con quattro
thread per job, LightPFN 1.1 su un AMD EPYC 9654 con 12 thread per job (otto job in parallelo) e su una RTX 5090,
TabICLv2 su una RX 7900 XT.

| Modello | Hardware | Tabelle piccole | Medie | Grandi | Predict, s per 1.000 righe |
|---|---|---:|---:|---:|---:|
| LightGBM | i7, 4 thread | 0,15 s | 0,32 s | 0,29 s | 0,004 |
| XGBoost | i7, 4 thread | 0,05 s | 0,16 s | 0,48 s | 0,015 |
| CatBoost | i7, 4 thread | 2,87 s | 4,25 s | 6,71 s | 0,018 |
| TabICLv2 | RX 7900 XT | 0,49 s | 3,20 s | 5,45 s | 2,87 |
| LightPFN 1.1, 4 estimatori | EPYC, 12 thread | 1,03 s | 15,4 s | 54,1 s | 4,12 |
| LightPFN 1.1, 1 estimatore | EPYC, 12 thread | 0,23 s | 3,87 s | 13,5 s | 1,05 |
| LightPFN 1.1, 4 estimatori | RTX 5090 | 0,02 s | 0,24 s | 1,18 s | 0,06 |

Le tabelle piccole hanno meno di 2.500 righe, le grandi 10.000 o più. Sulle tabelle grandi il costo su CPU è
l'attenzione di in-context learning su contesti lunghi; una GPU ne elimina la maggior parte. I tempi Vulkan sono
in [VULKAN.md](VULKAN.md).

**Implementazione dell'inferenza.** Il percorso di default (stadi sulle celle a blocchi per la cache, copia
"piegata" della rete, batch degli estimatori su GPU) è da 1,1 a 4,1 volte più veloce su CPU dell'implementazione
semplice, con memoria di picco fino a 2,3 volte più bassa e le stesse predizioni a meno degli arrotondamenti
(report della 1.1, Sezione 10). Nella 1.5, `precision="auto"` (fp16 su GPU CUDA/ROCm compatibili) e le fusioni
accelerano ulteriormente su GPU il train di 1,89x e il predict di 2,04x sul protocollo 8 fold x 4 membri.

## Feature categoriche: cosa non ha funzionato

La versione 1.1 legge le colonne categoriche come codici ordinali. Sulle tabelle grandi con categoriche ad alta
cardinalità (da centinaia a migliaia di livelli) le statistiche del target ordinate di CatBoost ricavano da
quelle colonne da 0,4 a 2 punti di AUC con 10.000 righe; LightPFN 1.1 quasi niente. Nella 1.1 abbiamo provato,
senza riaddestrare il modello:

- 13 codifiche a sola inferenza su D3 (frequenza, one-hot, ricodifica casuale, target encoding smussato e
  leave-one-out): nessuna dà un guadagno chiaro; il target encoding costa da 0,3 a 0,5 punti e le statistiche
  leave-one-out fanno collassare il modello (-12 punti);
- le colonne di conteggio di Kumo Tabular di NVIDIA (il logaritmo del conteggio per categoria, per colonne con
  più di 50 livelli) sui sei dataset di D4 con colonne di quel tipo: +0,03 punti [-0,19, 0,23], rumore;
- un adapter categorico (6.384 parametri: pesi di Fourier separati per le colonne categoriche e statistiche del
  target ordinate in stile CatBoost), addestrato con i pesi rilasciati della 1.1 congelati: nessun guadagno
  (da -0,06 a +0,01 punti).

Il meccanismo funziona, ma il prior v1 non contiene colonne categoriche come quelle dei dati reali (molti livelli
rari, ciascuno con un piccolo effetto proprio), quindi non c'era niente da imparare congelando i pesi della 1.1.
La versione 1.5 addestra quindi l'adattatore categorico da zero su un mix di prior con il 90% di `graph` v4 (che
include colonne categoriche ad alta cardinalità quantizzate) e `--cat-mask-drop 0.25`, guadagnando +3,1 punti di
AUC sui sei task Dcat a 10.000 righe rispetto alla baseline a 30.000 passi senza adattatore.
