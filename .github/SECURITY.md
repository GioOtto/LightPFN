# Security · Sicurezza

## English

### Reporting a vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private reporting under the **Security** tab, or
write to **giorgio.ottoboni@proton.me** with a minimal synthetic reproduction. State the LightPFN version,
the operating system, the device (`cpu`, `cuda`, `vulkan`) and the steps to reproduce the problem. Do not
attach private datasets, tokens or unreviewed logs.

The project is maintained by one person: there is no guaranteed response time, but serious reports are read
and addressed.

### Supported versions

Only the latest release on PyPI receives fixes.

### Scope and threat model

In scope: the `lightpfn` package (checkpoint loading, the scikit-learn wrapper, the PyTorch and Vulkan
backends) and the training code in this repository.

- **Checkpoints.** The package loads `model.safetensors` with a JSON config, or PyTorch files through
  `torch.load(weights_only=True)`. It never falls back to unrestricted pickle loading and never runs code
  downloaded from a model repository. A way to execute code through a checkpoint is a vulnerability.
- **Downloads.** Without `checkpoint=` or `model=`, the classifier downloads the weights from Hugging Face
  at a pinned commit. Custom repositories require an explicit revision. `local_files_only=True` forbids
  network access.
- **GPU drivers.** The Vulkan backend submits compute work through wgpu. Driver crashes or GPU resets caused
  by very large inputs are bugs; report them as ordinary issues unless they cross a security boundary.

Out of scope: vulnerabilities in PyTorch, wgpu, GPU drivers, Hugging Face Hub or other upstream projects,
which should be reported to those projects.

A classifier can be wrong. Do not use its predictions for decisions about people without validating them on
your own data.

## Italiano

### Segnalare una vulnerabilità

Non aprire una issue pubblica per una vulnerabilità. Usa la segnalazione privata nella scheda **Security** di
GitHub, oppure scrivi a **giorgio.ottoboni@proton.me** con una riproduzione sintetica e minima. Indica la
versione di LightPFN, il sistema operativo, il dispositivo (`cpu`, `cuda`, `vulkan`) e i passi per riprodurre
il problema. Non allegare dataset privati, token o log non controllati.

Il progetto è mantenuto da una persona sola: non ci sono tempi di risposta garantiti, ma le segnalazioni serie
vengono lette e affrontate.

### Versioni supportate

Riceve correzioni solo l'ultima release su PyPI.

### Perimetro e modello di minaccia

Rientrano nel perimetro il pacchetto `lightpfn` (caricamento dei checkpoint, wrapper scikit-learn, backend
PyTorch e Vulkan) e il codice di addestramento di questa repository.

- **Checkpoint.** Il pacchetto carica `model.safetensors` con una configurazione JSON, oppure file PyTorch con
  `torch.load(weights_only=True)`. Non ricade mai sul caricamento pickle senza restrizioni e non esegue mai
  codice scaricato da una repository di modelli. Un modo per eseguire codice tramite un checkpoint è una
  vulnerabilità.
- **Download.** Senza `checkpoint=` o `model=`, il classificatore scarica i pesi da Hugging Face a un commit
  fissato. Le repository personalizzate richiedono una revisione esplicita. `local_files_only=True` vieta
  l'accesso alla rete.
- **Driver GPU.** Il backend Vulkan invia il lavoro di calcolo tramite wgpu. Crash del driver o reset della GPU
  causati da input molto grandi sono bug; segnalali come issue normali, a meno che non superino un confine di
  sicurezza.

Restano fuori le vulnerabilità di PyTorch, wgpu, driver GPU, Hugging Face Hub e degli altri progetti a monte,
da segnalare ai rispettivi progetti.

Un classificatore può sbagliare. Non usare le sue predizioni per decisioni sulle persone senza validarle sui
tuoi dati.
