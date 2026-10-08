# Contributing to LightPFN · Contribuire a LightPFN

## English

Open an issue describing the problem or the change you have in mind, then a focused pull request with your
reasoning and the checks you ran. Small, reviewable changes are merged faster than large ones.

### Project rules

- **The model stays small.** Architecture changes keep the parameter count near 4.6M; LightPFN is meant to run
  on a commodity CPU.
- **No other foundation models in training.** No distillation, and no weights or predictions of other tabular
  foundation models in pretraining data or losses. Their papers are a fine source of ideas; their models may
  appear only as evaluation baselines.
- **Predictions stay exact.** A change to an inference path (cache blocking, folding, batching, Vulkan) must
  return the same probabilities as the plain PyTorch path up to floating-point rounding, with a test that
  checks it.
- **Evaluation discipline.** Model or prior changes are judged on held-out synthetic tasks and on real datasets
  outside TabArena; TabArena is kept for final checks. Report paired differences with intervals, not single
  numbers.
- **Licenses.** New dependencies need a license compatible with Apache 2.0, listed in `licenses/`. Keep the
  attributions of existing code.
- **Nothing private.** Do not add datasets, checkpoints, logs, credentials or paths from your machine.
- **AI tools.** Say in the pull request if you used AI tools, and review their output before submitting.

### Development setup

```bash
git clone https://github.com/GioOtto/LightPFN.git
cd LightPFN
pip install -e ".[dev]"                 # inference and tests
pip install -e ".[dev,vulkan,train,eval]"  # everything, including priors and the evaluation harness
```

### Checks before proposing

```bash
python -m pytest -q tests/test_release.py tests/test_sklearn_api.py tests/test_sklearn_context.py \
    tests/test_inference_paths.py tests/test_model.py tests/test_model_variants.py
LIGHTPFN_VULKAN_ADAPTER=llvmpipe python -m pytest -q tests/test_vulkan.py   # needs the vulkan extra and Mesa's llvmpipe
python -m pytest -q tests/                                                 # full suite, needs the train extra
```

The CI runs the inference tests on Linux and Windows with Python 3.10 and 3.12. Changes to `lightpfn/prior/`
or `lightpfn/train.py` should also pass `tests/test_prior.py`, `tests/test_data_path.py` and
`tests/test_stream.py`. If you change a figure of the report, regenerate it from its data in `paper/plotdata/`.

Contributions are offered under the project's Apache License 2.0.

## Italiano

Apri una issue per descrivere il problema o la modifica che hai in mente, poi una pull request circoscritta con
la motivazione e le verifiche eseguite. Le modifiche piccole e facili da rivedere entrano prima di quelle grandi.

### Regole del progetto

- **Il modello resta piccolo.** Le modifiche all'architettura mantengono il numero di parametri vicino a 4,6M;
  LightPFN deve girare su una CPU comune.
- **Niente altri modelli fondazionali nell'addestramento.** Niente distillazione, niente pesi o predizioni di
  altri modelli fondazionali tabellari nei dati di preaddestramento o nelle loss. I loro paper sono una buona
  fonte di idee; i loro modelli possono comparire solo come baseline di valutazione.
- **Le predizioni restano esatte.** Una modifica a un percorso di inferenza (cache blocking, folding, batching,
  Vulkan) deve restituire le stesse probabilità del percorso PyTorch semplice a meno degli arrotondamenti in
  virgola mobile, con un test che lo verifica.
- **Disciplina di valutazione.** Le modifiche a modello o prior si giudicano su problemi sintetici tenuti da
  parte e su dataset reali esterni a TabArena; TabArena resta per le verifiche finali. Riporta differenze
  appaiate con intervalli, non numeri singoli.
- **Licenze.** Le nuove dipendenze devono avere una licenza compatibile con Apache 2.0 ed essere elencate in
  `licenses/`. Conserva le attribuzioni del codice esistente.
- **Niente di privato.** Non aggiungere dataset, checkpoint, log, credenziali o percorsi della tua macchina.
- **Strumenti AI.** Indica nella pull request se hai usato strumenti AI e controllane l'output prima di proporla.

### Ambiente di sviluppo

```bash
git clone https://github.com/GioOtto/LightPFN.git
cd LightPFN
pip install -e ".[dev]"                 # inferenza e test
pip install -e ".[dev,vulkan,train,eval]"  # tutto, compresi prior e harness di valutazione
```

### Verifiche prima di proporre

```bash
python -m pytest -q tests/test_release.py tests/test_sklearn_api.py tests/test_sklearn_context.py \
    tests/test_inference_paths.py tests/test_model.py tests/test_model_variants.py
LIGHTPFN_VULKAN_ADAPTER=llvmpipe python -m pytest -q tests/test_vulkan.py   # serve l'extra vulkan e llvmpipe di Mesa
python -m pytest -q tests/                                                 # suite completa, serve l'extra train
```

La CI esegue i test di inferenza su Linux e Windows con Python 3.10 e 3.12. Le modifiche a `lightpfn/prior/` o a
`lightpfn/train.py` devono passare anche `tests/test_prior.py`, `tests/test_data_path.py` e
`tests/test_stream.py`. Se cambi una figura del report, rigenerala dai dati in `paper/plotdata/`.

I contributi vengono proposti sotto la licenza Apache 2.0 del progetto.

---

Contact · Contatto: **giorgio.ottoboni@proton.me**
