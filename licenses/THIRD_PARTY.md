# Third-party licenses · Licenze di terze parti

LightPFN's code and weights are under the Apache License 2.0 (see `LICENSE` and `NOTICE`). The repository and
the published wheel contain no third-party source code or weights: the projects below are installed by pip as
dependencies and keep their own licenses.

Il codice e i pesi di LightPFN sono sotto Apache License 2.0 (vedi `LICENSE` e `NOTICE`). La repository e il
wheel pubblicato non contengono codice o pesi di terze parti: i progetti qui sotto vengono installati da pip
come dipendenze e mantengono le proprie licenze.

| Package | Used for / Uso | License / Licenza |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | model, CPU/CUDA/ROCm inference, training | BSD-3-Clause (with bundled components under Apache-2.0, MIT and others) |
| [NumPy](https://github.com/numpy/numpy) | arrays | BSD-3-Clause |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | estimator API, input validation | BSD-3-Clause |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | weight download at a pinned commit | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | weight format | Apache-2.0 |
| [wgpu-py](https://github.com/pygfx/wgpu-py) (extra `vulkan`) | Vulkan backend; bundles wgpu-native | BSD-2-Clause; wgpu-native MIT or Apache-2.0 |
| [TabICL](https://github.com/soda-inria/tabicl) (extra `train`) | structural causal model prior used by `lightpfn.prior` | BSD-3-Clause |
| [pandas](https://github.com/pandas-dev/pandas) (extras `train`, `eval`, `dev`; optional DataFrame input) | data handling | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy) (also a scikit-learn dependency) | prior sampling, estimator support | BSD-3-Clause |
| [threadpoolctl](https://github.com/joblib/threadpoolctl), [joblib](https://github.com/joblib/joblib) | thread limits, parallel evaluation | BSD-3-Clause |
| [openml-python](https://github.com/openml/openml-python) (extra `eval`) | evaluation datasets | BSD-3-Clause |
| [CatBoost](https://github.com/catboost/catboost) (extra `eval`) | baseline | Apache-2.0 |
| [LightGBM](https://github.com/microsoft/LightGBM) (extra `eval`) | baseline | MIT |
| [XGBoost](https://github.com/dmlc/xgboost) (extra `eval`) | baseline | Apache-2.0 |
| [pytest](https://github.com/pytest-dev/pytest) (extra `dev`) | tests | MIT |
| [build](https://github.com/pypa/build) (extra `dev`) | package builds | MIT |
| [Hatchling](https://github.com/pypa/hatch) (build dependency) | wheel and source distribution backend | MIT |

Datasets used in the evaluations (OpenML, TabArena) are downloaded at run time from their sources and are not
redistributed. Other tabular foundation models (TabICLv2 and others) appear in the report only as evaluation
baselines; their weights and outputs were never used in training.

I dataset usati nelle valutazioni (OpenML, TabArena) vengono scaricati al momento dell'esecuzione dalle loro
fonti e non sono ridistribuiti. Gli altri modelli fondazionali tabellari (TabICLv2 e altri) compaiono nel report
solo come baseline di valutazione; i loro pesi e output non sono mai stati usati nell'addestramento.
