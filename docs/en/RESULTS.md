# Results

**English** · [Italiano](../it/RISULTATI.md)

Protocols are described in the [technical report](../../paper/LightPFN_report.pdf). The tables report results
from the report and the linked CSVs. The official TabArena-Lite section includes default, tuned and ensembled
baselines; the other comparisons use default baselines. AUC differences are in points (100 times the AUC difference), with 95% paired bootstrap
intervals over tasks or datasets. The released model is the long-context checkpoint (LC39k in the report).

- [Evaluation sets](#evaluation-sets)
- [Small tables: 55 OpenML datasets outside TabArena](#small-tables-55-openml-datasets-outside-tabarena)
- [TabArena, official pipeline](#tabarena-official-pipeline)
- [TabArena, 38 classification tasks with our harness](#tabarena-38-classification-tasks-with-our-harness)
- [Larger tables](#larger-tables)
- [Cost](#cost)
- [Categorical features: what did not work](#categorical-features-what-did-not-work)

## Evaluation sets

Design decisions were taken on three sets kept apart from TabArena: D1, 1,536 held-out synthetic tasks; D2,
mechanism probes (XOR, parity, lookup tables among noise columns); D3, 55 OpenML-CC18 classification datasets
that are not in TabArena. For this series, after introducing the selection protocol, TabArena was looked at
only for the two finalists, after choosing between them. An earlier exploratory r2 evaluation predates this
protocol and is disclosed in the technical report, Section 7.

## Small tables: 55 OpenML datasets outside TabArena

At most 1,000 rows and 100 random features per dataset, stratified five-fold cross-validation, one estimator.

| Model | Mean AUC | LightPFN minus model | LightPFN higher on |
|---|---:|---:|---:|
| **LightPFN** | **0.9109** | | |
| CatBoost | 0.9023 | +0.86 [0.42, 1.40] | 80% of datasets |
| Random forest | 0.8933 | +1.76 [1.03, 2.62] | 91% |
| LightGBM | 0.8907 | +2.02 [1.14, 3.16] | 91% |
| XGBoost, 54 datasets | 0.8889 | +2.06 [1.32, 2.94] | 94% |

XGBoost's categorical wrapper fails on one dataset (*sick*), which is excluded from both sides of its comparison.
The lead over CatBoost holds on binary (+0.80) and multiclass (+0.92) datasets, both with intervals above zero.

[D3 comparison CSV](../assets/d3_baselines.csv), including bootstrap intervals and win fractions.

## TabArena, official pipeline

Author-run evaluation on 8 October 2026 with `TabArenaV0pt1ExperimentBundle(models=[("LightPFN", 0)])` and
`subset=["lite", "classification"]`: first official split of each of the 38 classification datasets,
eight-fold bagging, official `8x1` validation protocol. Default configuration, four estimators, one checkpoint,
no HPO search space. Context caching is off in fold models and on in the refit model. All 38 tasks succeeded,
none imputed. The 13 regression datasets are excluded.

**25th of 99 methods, Elo 1420 (+67 / -66)**; mean rank 38.37, harmonic rank 17.37, score 0.270.
25th is the position sorted by Elo; mean rank is a separate metric computed over datasets.

| Method | Elo | 95% CI |
|---|---:|---:|
| Kumo-Tabular (default) | 1995 | +163 / -119 |
| LimiX-2 (default) | 1860 | +169 / -109 |
| Mitra-v2 (default) | 1727 | +134 / -95 |
| TabPFN-2.6 (default) | 1559 | +60 / -53 |
| TabICLv2 (default) | 1558 | +77 / -68 |
| TabDPT-1.3 (default) | 1467 | +78 / -54 |
| RealMLP (tuned + ensembled) | 1459 | +50 / -47 |
| **LightPFN (default)** | **1420** | **+67 / -66** |
| RealMLP (tuned) | 1403 | +54 / -57 |
| CatBoost (tuned) | 1378 | +58 / -55 |
| CatBoost (tuned + ensembled) | 1370 | +58 / -48 |
| LightGBM (tuned + ensembled) | 1365 | +52 / -42 |
| XGBoost (tuned + ensembled) | 1346 | +58 / -62 |
| TabICL (default) [5.26% imputed] | 1343 | +72 / -65 |
| CatBoost (default) | 1339 | +49 / -52 |
| XGBoost (default) | 1191 | +57 / -69 |
| LightGBM (tuned) | 1316 | +52 / -47 |
| XGBoost (tuned) | 1320 | +64 / -62 |
| LightGBM (default) | 1144 | +59 / -63 |
| RealMLP (default) | 1247 | +64 / -53 |
| RandomForest (default) | 1000 | +71 / -84 |

The Elo point estimate is above every GBDT in the comparison, including tuned and ensembled entries. The
intervals with tuned CatBoost (1378, +58 / -55) overlap and do not establish a clear win. TabICLv2 and larger
foundation models remain ahead. TabICL v1 has 5.26% imputed results; LightPFN has none.

![Elo against fit time, LightPFN in purple](../assets/tabarena_lite_pareto.png)

LightPFN hardware: Intel i7-13700KF (16 cores, 24 threads), 32 GB RAM, AMD RX 7900 XT 20 GB, Linux,
ROCm 7.2, PyTorch 2.13.0, Python 3.12.15. Median fit 0.65 s and median predict 0.086 s per 1,000 rows.
Only TabDPT-1.3 reports a faster fit among entries with at least its Elo; baseline timings are TabArena's
published measurements on different hardware, so this is not a controlled speed comparison.

[Leaderboard CSV](../assets/tabarena_lite_leaderboard.csv), [entry-point script and full results (ZIP)](https://github.com/GioOtto/LightPFN/releases/download/v1.0.0/LightPFN-1.0.0-tabarena-lite.zip).
The script imports `torch` before AutoGluon so it detects the AMD GPU through ROCm. TabArena maintainers re-run
the full benchmark on their hardware before leaderboard inclusion; this Lite position is not a final
maintainer-verified result.

## TabArena, 38 classification tasks with our harness

The 38 classification tasks of TabArena v0.1, official OpenML splits, first repeat (114 splits). This is our own
harness, not the official leaderboard protocol (which bags models, tunes them and ranks by Elo); the 13
regression tasks are not evaluated. Error is 1 - AUC for binary tasks and the log loss for multiclass tasks, as
in TabArena; ranks are computed on the error.

| Model | Mean AUC | Mean rank of 7 | Lower error than CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parameters, GPU) | 0.8637 | 1.58 | 87% |
| **LightPFN, 4 estimators** | **0.8581** | **2.50** | **76%** |
| LightPFN, 1 estimator | 0.8568 | 3.37 | 68% |
| CatBoost | 0.8553 | 3.50 | |
| LightGBM | 0.8440 | 5.34 | 5% |
| Random forest | 0.8383 | 5.89 | 3% |
| XGBoost | 0.8332 | 5.82 | 11% |

The mean AUC difference from CatBoost is +0.28 points [-0.26, 0.80]: LightPFN wins more tasks, but the mean
difference is not significant. TabICLv2 is ahead by 0.56 points [0.25, 0.98].

By group (LightPFN with 4 estimators against CatBoost):

| Group | Tasks | Mean rank, LightPFN / CatBoost / TabICLv2 | AUC points vs CatBoost | Tasks won |
|---|---:|---|---:|---:|
| binary | 30 | 2.60 / 3.30 / 1.57 | +0.25 | 70% |
| multiclass | 8 | 2.13 / 4.25 / 1.63 | +0.38 | 100% |
| fewer than 2,500 rows | 12 | 2.25 / 4.25 / 1.42 | +1.18 | 100% |
| 2,500 to 10,000 rows | 10 | 2.70 / 3.20 / 1.80 | +0.28 | 60% |
| 10,000 rows or more | 16 | 2.56 / 3.13 / 1.56 | -0.40 | 69% |
| minority class 1 to 10% | 11 | 2.55 / 2.91 / 1.82 | -0.35 | 64% |

The largest losses are on mostly categorical tables: *in_vehicle_coupon_recommendation* (-4.49 points),
*Amazon_employee_access* (-3.95) and *Diabetes130US* (-2.35).

<details>
<summary>Mean AUC per task (best in bold), sorted by LightPFN minus CatBoost</summary>

| Task | Train rows | Features | Classes | LightPFN x4 | CatBoost | TabICLv2 | LightGBM | XGBoost | RF |
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

[Per-task AUC CSV from our harness](../assets/tabarena_harness.csv).

## Larger tables

**Learning curves.** The 15 D3 datasets with at least 5,000 rows, two repeats, the whole training set as context
(one estimator). Difference from default CatBoost:

| Training rows | Datasets | CatBoost AUC | LightPFN minus CatBoost |
|---|---:|---:|---:|
| 1,000 | 15 | 0.8959 | +0.46 [0.13, 0.95] |
| 2,000 | 15 | 0.9071 | +0.51 [0.22, 0.84] |
| 4,000 | 15 | 0.9187 | +0.33 [-0.08, 0.74] |
| 8,000 | 9 | 0.9085 | +0.15 [-0.42, 0.74] |
| 16,000 | 6 | 0.8723 | -0.38 [-1.70, 0.92] |
| 32,000 | 6 | 0.8784 | -0.42 [-1.41, 0.70] |
| all (4,324 to 94,320) | 15 | 0.9313 | +0.31 [-0.23, 0.86] |

**Large datasets (D4).** 13 OpenML datasets of 50,000 to 2.2 million rows outside TabArena and D3, 5,000 test
rows, one repeat, the whole training set as context:

| Training rows | Datasets | CatBoost AUC | LightPFN AUC | LightPFN minus CatBoost | LightGBM / XGBoost minus CatBoost |
|---|---:|---:|---:|---:|---:|
| 10,000 | 13 | 0.8141 | 0.8168 | +0.28 [-0.53, 1.20] | -4.33 / -3.30 |
| 25,000 | 13 | 0.8225 | 0.8234 | +0.09 [-0.81, 1.08] | -3.52 / -2.72 |
| 50,000 | 12 | 0.8311 | 0.8291 | -0.20 [-1.17, 0.76] | -4.01 / -2.78 |
| 100,000 | 8 | 0.8027 | 0.7993 | -0.34 [-1.95, 1.29] | -0.83 / -2.51 |

LightPFN leads on *porto-seguro*, *jannis*, *covertype* and *Higgs* and trails on *albert*, *road-safety* and *kick*,
where high-cardinality categorical columns matter.

## Cost

**TabArena, median fit plus predict time per split** (full protocol). The hosts differ, so the comparison across
rows is indicative only: tree ensembles ran on an Intel i7-13700KF with four threads per job, LightPFN on an AMD
EPYC 9654 with 12 threads per job (eight concurrent jobs) and on one RTX 5090, TabICLv2 on an RX 7900 XT.

| Model | Hardware | Small tables | Medium | Large | Predict, s per 1,000 rows |
|---|---|---:|---:|---:|---:|
| LightGBM | i7, 4 threads | 0.15 s | 0.32 s | 0.29 s | 0.004 |
| XGBoost | i7, 4 threads | 0.05 s | 0.16 s | 0.48 s | 0.015 |
| CatBoost | i7, 4 threads | 2.87 s | 4.25 s | 6.71 s | 0.018 |
| TabICLv2 | RX 7900 XT | 0.49 s | 3.20 s | 5.45 s | 2.87 |
| LightPFN, 4 estimators | EPYC, 12 threads | 1.03 s | 15.4 s | 54.1 s | 4.12 |
| LightPFN, 1 estimator | EPYC, 12 threads | 0.23 s | 3.87 s | 13.5 s | 1.05 |
| LightPFN, 4 estimators | RTX 5090 | 0.02 s | 0.24 s | 1.18 s | 0.06 |

Small tables have fewer than 2,500 rows, large ones 10,000 or more. On large tables the CPU cost is the
in-context attention over long contexts; a GPU removes most of it. Vulkan timings are in [VULKAN.md](VULKAN.md).

**Inference implementation.** The default path (cache-blocked cell stages, a folded copy of the network,
estimator batching on GPU) is 1.1 to 4.1 times faster on CPU than the plain implementation, with peak memory up
to 2.3 times lower and the same predictions up to rounding (report, Section 10).

## Categorical features: what did not work

Version 1 reads categorical columns as ordinal codes. On large tables with high-cardinality categorical columns
(hundreds to thousands of levels) CatBoost's ordered target statistics extract 0.4 to 2 AUC points from those
columns at 10,000 rows; LightPFN extracts almost nothing. We tried, without retraining the model:

- 13 inference-only encodings on D3 (frequency, one-hot, random recoding, smoothed target and leave-one-out
  encodings): none gives a clear gain; target encodings cost 0.3 to 0.5 points and leave-one-out statistics
  make the model collapse (-12 points);
- the count columns of NVIDIA's Kumo Tabular (a log count per category, for columns with more than 50 levels)
  on the six D4 datasets with such columns: +0.03 points [-0.19, 0.23], noise;
- a categorical adapter (6,384 parameters: separate Fourier weights for categorical columns and CatBoost-style
  ordered target statistics), fine-tuned with the released weights frozen: no gain (-0.06 to +0.01 points).

The mechanism works, but the v1 prior has no categorical columns of the kind real data has (many rare levels,
each with a small effect of its own), so there is nothing to learn. Native categorical handling therefore moves
to version 2, together with a prior that contains such columns.
