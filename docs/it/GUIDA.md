# Guida all'uso

[English](../en/GUIDE.md) · **Italiano**

Gli esempi rapidi di classificazione e regressione sono nel [README](../../README.it.md). Sia
`LightPFNClassifier` sia `LightPFNRegressor` espongono `fit`, `predict`, `score`, `get_params` e `set_params`, e
supportano `clone`, `Pipeline` e `GridSearchCV`. Il classificatore aggiunge `predict_proba`; il regressore
aggiunge `predict_quantiles(X, quantiles=None)`.

- [Cosa fa un fit](#cosa-fa-un-fit)
- [Regressione](#regressione)
- [Parametri](#parametri)
- [Attributi dopo il fit](#attributi-dopo-il-fit)
- [Dati in ingresso](#dati-in-ingresso)
- [Categoriche](#categoriche)
- [Tabelle grandi](#tabelle-grandi)
- [Dispositivi](#dispositivi)
- [Precisione](#precisione)
- [Velocità](#velocità)
- [Pesi, uso offline e checkpoint propri](#pesi-uso-offline-e-checkpoint-propri)
- [Riproducibilità](#riproducibilità)
- [API di basso livello](#api-di-basso-livello)

## Cosa fa un fit

`fit(X, y, cat="auto")` non addestra nulla. Valida i dati, carica la rete preaddestrata (una volta per istanza
dell'estimatore) e codifica il training set: statistiche delle colonne, statistiche categoriche quando
l'adattatore categorico è attivo, stati dei punti induttori dei due stadi per colonna e chiavi e valori dei
blocchi di in-context learning. `predict_proba(X)` (oppure `predict` / `predict_quantiles` in regressione) fa
passare nella rete solo le righe di test, a blocchi, e le fa guardare quel contesto in cache. La predizione di
una riga di test non dipende mai dalle altre righe di test.

Questo descrive il default `cache_context=True`. Con `cache_context=False` (entrambi gli estimatori), `fit`
conserva copie degli input; ogni predizione costruisce lo stesso contesto e poi lo libera. Su CUDA/ROCm, quando
righe di training e di test stanno in un solo blocco, quella predizione codifica il contesto e predice le righe
di test in un unico passaggio che condivide tra le due tutte le operazioni per riga (circa metà dei lanci di
kernel GPU): la stessa funzione a meno degli arrotondamenti. È adatto ai modelli dei fold che predicono una
volta sola, come nel bagging.

In `LightPFNClassifier` con `n_estimators > 1`, il primo estimatore usa l'ordine originale delle feature e degli
slot di classe, ogni altro una permutazione casuale delle feature e un'assegnazione casuale delle classi agli
slot di etichetta del modello; le probabilità vengono mediate. Su TabArena (1.1), quattro estimatori alzano
l'AUC media di circa 0,13 punti rispetto a uno e migliorano il rank medio su sette modelli da 3,37 a 2,50, a un
costo quattro volte maggiore su CPU.

## Regressione

`LightPFNRegressor` è un regressore quantile in-context preaddestrato da 4.772.519 parametri. Il target di
regressione `y` deve essere un vettore 1-D finito di forma `(n_righe,)`; matrici di target multioutput non sono
supportate.

- `predict_quantiles(X, quantiles=None)` restituisce un array `(n_righe, 999)` per i 999 livelli quantili fissi
  da `.001` a `.999` quando `quantiles=None`, oppure interpola i livelli richiesti in `[0, 1]`.
- `predict(X)` restituisce la media predittiva integrando la curva dei 999 quantili con la regola dei trapezi.
- Con il default `n_estimators=2` (valori ammessi `1` o `2`), il primo membro codifica `y` nelle unità
  originali. Controllato da `log_skew=2.0`, il secondo membro codifica `log(y)` solo se tutti i target del
  contesto sottocampionato sono strettamente positivi e l'asimmetria campionaria corretta di Fisher-Pearson
  (come in `pandas.Series.skew()`) è almeno pari a `log_skew`; altrimenti il secondo membro codifica `-y`
  (`"flip"`). Impostare `log_skew=None` disabilita la trasformazione logaritmica e usa sempre `-y` per il
  secondo membro. L'attributo `second_` registra `"log"`, `"flip"` oppure `None` con un solo membro.
- I quantili del membro `"log"` vengono riportati alle unità originali con `exp` e limitati all'intervallo del
  target nel contesto allargato di un intervallo per lato, con limite inferiore zero. I quantili del membro
  `"flip"` vengono cambiati di segno e invertiti di ordine. I due membri vengono mediati per livello quantile e
  riordinati.
- `query_rows=4096` limita il numero di righe di test per blocco di predizione in regressione;
  `chunk_cells="auto"` limita il lavoro degli stadi sulle celle nel frontend.

## Parametri

### Riepilogo

| Parametro | Default in `LightPFNClassifier` | Default in `LightPFNRegressor` |
|---|---|---|
| `n_estimators` | `4` (permutazioni di feature e slot di classe) | `2` (`1` o `2`: `y` più `log(y)` o `-y`) |
| `max_context` | `20000` (sottoinsieme stratificato per membro) | `50000` (sottoinsieme uniforme) |
| `random_state` | `None` (alias `seed=0`) | `0` |
| `device` | `"auto"` (`cuda`, `vulkan`, `cpu`) | `"auto"` (`cuda`, `vulkan`, `cpu`) |
| `precision` | `"auto"` (`"auto"`, `"fp32"`, `"fp16"`) | `"auto"` (`"auto"`, `"fp32"`, `"fp16"`) |
| `checkpoint` | `None` (file `.pt` o cartella safetensors locale) | `None` (file `.pt` o cartella safetensors locale) |
| `repo_id`, `revision`, `cache_dir`, `local_files_only` | opzioni Hugging Face | opzioni Hugging Face |
| `n_threads` | `None` (thread CPU di PyTorch per il processo) | `None` (thread CPU di PyTorch per il processo) |

### Parametri del classificatore

```python
LightPFNClassifier(model=None, checkpoint=None, device="auto", n_estimators=4, max_context=20000,
                   chunk_rows=2048, n_threads=None, seed=0, *, chunk_cells="auto", batch_cells="auto",
                   fold=True, random_state=None, repo_id=None, revision=None, cache_dir=None,
                   local_files_only=False, cache_context=True, precision="auto")
```

| Parametro | Default | Significato |
|---|---|---|
| `n_estimators` | `4` | estimatori mediati su permutazioni di feature e slot di etichetta; 1 è circa quattro volte più veloce su CPU |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda[:i]"`, `"vulkan[:i]"`; vedi [Dispositivi](#dispositivi) |
| `precision` | `"auto"` | `"auto"` (fp16 su GPU CUDA/ROCm compatibili da 2.048 celle di contesto, fp32 su CPU e Vulkan), `"fp32"` o `"fp16"`; vedi [Precisione](#precisione) |
| `max_context` | `20000` | oltre questo numero di righe di training, ogni estimatore legge un sottoinsieme stratificato di questa dimensione |
| `random_state` | `None` | seme di permutazioni e sottoinsiemi (`seed` è il vecchio alias; impostane uno solo) |
| `n_threads` | `None` | thread CPU di PyTorch (`torch.set_num_threads`, vale per tutto il processo); `None` lascia l'impostazione di PyTorch |
| `chunk_rows` | `2048` | righe di test per blocco di predizione (limita la memoria) |
| `checkpoint` | `None` | un `model.safetensors` locale (con `config.json` accanto), la sua cartella o un file `.pt` |
| `model` | `None` | un modulo torch `LightPFN` da usare al posto dei pesi scaricati (copiato al fit) |
| `repo_id`, `revision` | `None` | un'altra repository o revisione Hugging Face; una repository personalizzata richiede una revisione esplicita |
| `cache_dir` | `None` | cartella della cache Hugging Face |
| `local_files_only` | `False` | non usare mai la rete; i pesi devono essere già in cache |
| `cache_context` | `True` | conserva i contesti codificati tra le predizioni; `False` conserva copie degli input e ricostruisce e libera i contesti a ogni predizione, utile per i modelli dei fold usati una volta sola |
| `chunk_cells`, `batch_cells`, `fold` | `"auto"`, `"auto"`, `True` | implementazione dell'inferenza: cache blocking, batch degli estimatori su GPU e rete "piegata". Cambiano velocità e memoria, non le predizioni (a meno degli arrotondamenti). `chunk_cells=None, batch_cells=0, fold=False` è il percorso semplice |

### Parametri del regressore

```python
LightPFNRegressor(model=None, checkpoint=None, device="auto", precision="auto", n_estimators=2,
                  max_context=50000, random_state=0, log_skew=2.0, query_rows=4096, chunk_cells="auto",
                  n_threads=None, repo_id=None, revision=None, cache_dir=None, local_files_only=False,
                  fold=False, cache_context=True)
```

| Parametro | Default | Significato |
|---|---|---|
| `n_estimators` | `2` | `1` o `2`; con `2`, media `y` con `log(y)` oppure `-y` nello spazio dei quantili |
| `device` | `"auto"` | `"auto"` (CUDA/ROCm se disponibile, altrimenti CPU), `"cpu"`, `"cuda[:i]"` |
| `precision` | `"auto"` | `"auto"`, `"fp32"` o `"fp16"`; vedi [Precisione](#precisione) |
| `max_context` | `50000` | oltre questo numero di righe di training, estrae un sottoinsieme uniforme di `max_context` righe senza reinserimento |
| `random_state` | `0` | seme per il sottocampionamento del contesto |
| `log_skew` | `2.0` | soglia di asimmetria per usare `log(y)` come secondo membro su target positivi; `None` usa sempre `-y` |
| `query_rows` | `4096` | righe di test per blocco di predizione |
| `chunk_cells` | `"auto"` | celle per blocco degli stadi iniziali |
| `n_threads` | `None` | thread CPU di PyTorch (`torch.set_num_threads`, vale per tutto il processo); `None` lascia l'impostazione di PyTorch |
| `cache_context` | `True` | conserva il contesto codificato tra le predizioni; `False` conserva gli input e codifica a ogni predizione (in un solo passaggio su CUDA/ROCm per le tabelle piccole), utile per i modelli dei fold usati una volta sola |
| `fold` | `False` | usa la rete di inferenza "piegata" (stessa funzione a meno degli arrotondamenti) |
| `checkpoint`, `model`, `repo_id`, `revision`, `cache_dir`, `local_files_only` | `None` / `False` | checkpoint locale, modulo `LightPFN` personalizzato o opzioni di download da Hugging Face |

Entrambi i costruttori memorizzano i parametri e nient'altro, quindi `clone`, `get_params` e `set_params`
funzionano come in qualsiasi estimatore scikit-learn e non scaricano nulla né toccano la GPU.

## Attributi dopo il fit

### Attributi del classificatore dopo il fit

| Attributo | Contenuto |
|---|---|
| `classes_` | le etichette delle classi, nell'ordine delle colonne di `predict_proba` |
| `n_features_in_`, `feature_names_in_` | numero e nomi (per DataFrame con nomi di colonna stringa) delle feature |
| `is_categorical_` | maschera booleana delle colonne codificate come categorie da un DataFrame pandas |
| `cat_` | maschera booleana passata all'adattatore categorico (`None` se nessuna colonna è marcata o il modello caricato non ha l'adattatore) |
| `categories_` | dizionario dalla posizione della colonna al vocabolario categorico del fit; un dtype category conserva tutti i livelli dichiarati, anche quelli inutilizzati |
| `device_` | il dispositivo del backend selezionato, aggiornato da `to()` |
| `fit_device_` | il dispositivo che esegue davvero l'inferenza, compreso un eventuale ripiego automatico sulla CPU |
| `precision_` | `"fp16"` o `"fp32"`, la modalità di precisione scelta durante il `fit` |
| `n_precision_fallbacks_` | numero cumulativo di blocchi fp16 non finiti rieseguiti in fp32 a partire dal `fit` |
| `model_` | la rete `LightPFN` caricata |

### Attributi del regressore dopo il fit

| Attributo | Contenuto |
|---|---|
| `n_features_in_`, `feature_names_in_` | numero e nomi (per DataFrame con nomi di colonna stringa) delle feature |
| `device_` | il dispositivo torch selezionato |
| `second_` | `"log"`, `"flip"` oppure `None` (quando `n_estimators=1`), indica la trasformazione del secondo membro |
| `precision_` | `"fp16"` o `"fp32"`, la modalità di precisione scelta durante il `fit` |
| `n_precision_fallbacks_` | numero cumulativo di blocchi fp16 non finiti rieseguiti in fp32 a partire dal `fit` |
| `regressor_` | il motore `V15Regressor` sottostante (contiene `cat_`, `categories_`, `skew_`, `y_range_`, `ctx_`) |

## Dati in ingresso

- **Array**: qualsiasi array numerico di forma `(n_righe, n_feature)`. NaN indica un valore mancante; valori
  infiniti e matrici sparse vengono rifiutati.
- **DataFrame pandas**: le colonne di tipo category, stringa, object o bool diventano codici ordinali delle
  categorie viste in `fit` (nell'ordine di pandas: valori ordinati per le stringhe, l'ordine del dtype per un
  dtype category, compresi i livelli dichiarati inutilizzati). I valori mancanti e i valori fuori da quel
  vocabolario diventano NaN. Le colonne numeriche, comprese le nullable `Int64` e `Float64`, diventano float con
  `pd.NA` come NaN. In predizione le colonne devono avere gli stessi nomi e lo stesso ordine del `fit`.
- **Etichette di classificazione**: qualsiasi etichetta accettata da scikit-learn (interi, stringhe). Il
  classificatore è stato addestrato su 2-10 classi; oltre 10 viene sollevato un errore. Con una sola classe,
  `predict_proba` restituisce 1 per quella classe.
- **Target di regressione**: un vettore numerico 1-D finito di forma `(n_righe,)`. Valori NaN o infiniti nel
  target e matrici 2-D multioutput sollevano un `ValueError`.
- **Numero di feature**: i modelli sono stati addestrati fino a 100 feature e funzionano anche con di più; le
  tabelle larghe costano di più (gli stadi sulle celle sono lineari nel numero di feature).
- **Scalatura**: non serve. Ogni colonna viene standardizzata e trasformata in ranghi sulle righe di training,
  dentro il modello.

## Categoriche

Sia `LightPFNClassifier.fit(X, y, cat="auto")` sia `LightPFNRegressor.fit(X, y, cat="auto")` accettano
l'argomento `cat`:

- `cat="auto"` (predefinito): quando la rete caricata include l'adattatore categorico (`Config.cat_adapter=True`,
  attivo nei pesi predefiniti della 1.5), `auto_categorical(X)` marca i dtype pandas `category`, stringa,
  `object` e `bool`, più le colonne numeriche i cui valori non mancanti nel training set sono tutti interi con da
  2 a 20 livelli distinti. Il rilevamento legge solo le righe di training.
- Maschera booleana esplicita: un vettore booleano 1-D di lunghezza `n_features` marca direttamente le colonne
  categoriche.
- `cat=None`: in `LightPFNClassifier` conserva la marcatura basata sui dtype del DataFrame (`is_categorical_`)
  senza ispezionare le colonne numeriche intere; in `LightPFNRegressor` disabilita la maschera delle categoriche.

Le colonne marcate come categoriche vengono elaborate dall'adattatore categorico della 1.5 (pesi di Fourier
separati e statistiche ordinate del target). Nei pesi della 1.1 (`Config.cat_adapter=False`), le colonne
categoriche vengono lette solo come codici ordinali; passare una maschera `cat` esplicita a un modello privo di
adattatore emette un `FutureWarning`. Per tabelle dominate da colonne con migliaia di livelli rari, i modelli ad
alberi con statistiche del target dedicate come CatBoost restano spesso più forti.

## Tabelle grandi

L'attenzione di in-context learning cresce con il quadrato della lunghezza del contesto, quindi tempo e memoria
crescono in fretta con il numero di righe di training.

- **Classificazione (`max_context=20000`):** fino a `max_context` righe, ogni estimatore legge tutto il training
  set. Oltre, ogni estimatore legge un proprio sottoinsieme stratificato di `max_context` righe che tiene ogni
  classe (almeno cinque righe ciascuna, se conteggi e budget del contesto lo permettono). Più estimatori coprono
  allora una parte maggiore dei dati.
- **Regressione (`max_context=50000`):** oltre `max_context` righe, `fit` estrae un sottoinsieme uniforme di
  `max_context` righe senza reinserimento usando `random_state`.
- `max_context` si può alzare. Entrambi i checkpoint della 1.5 sono stati addestrati con uno stadio a contesto
  lungo fino a 60.000 righe. Nel report della 1.1 sono stati valutati contesti fino a 100.000 righe in
  classificazione, dove il modello è in media alla pari con CatBoost di default; il costo su CPU è alto (circa
  75 s per 94.000 righe con un estimatore su 24 thread).

## Dispositivi

<a id="precisione-e-dispositivi"></a>

`LightPFNClassifier` e `LightPFNRegressor` usano la stessa scelta del dispositivo: `device="auto"` prova una
GPU CUDA o ROCm tramite PyTorch (`torch.cuda.is_available()`), una GPU con driver Vulkan se `wgpu` è
installato, poi la CPU. La variabile d'ambiente `LIGHTPFN_DEVICE` sostituisce `"auto"` (per esempio
`LIGHTPFN_DEVICE=cpu`). Un dispositivo esplicito viene usato così com'è, con un errore se non è disponibile.

**Supporto e limiti di Vulkan nella 1.5:**

- Entrambi gli estimatori supportano colonne numeriche e categoriche marcate. Embedding delle celle,
  proiezioni delle statistiche categoriche, stadi di colonna e riga, ICL e attenzione del decoder girano sulla
  GPU. Statistiche di colonne/categorie e ricerca del rango restano sull'host. Anche la normalizzazione del
  target di regressione e la lettura finale della testa quantile girano sull'host. Vulkan usa sempre fp32.
- Su una RX 7900 XT, le probabilità di classificazione coincidono con torch CPU entro 2.6e-5 con argmax
  identico, su contesti fino a 12.000 righe con categoriche ad alta cardinalità. I quantili di regressione
  coincidono entro 5.5e-5 in relativo, su contesti fino a 40.000 righe. I casi 1.5 misurati con contesti più
  grandi sono da 3 a 7 volte più veloci della CPU; le tabelle piccole possono essere più lente. I driver Vulkan
  Intel e NVIDIA usano lo stesso codice ma non sono stati misurati.
- Su Vulkan, le celle di training di un estimatore (righe per feature per 64 float) devono stare in un singolo
  buffer della GPU (2 GiB sul driver della RX 7900 XT provato, circa 8,4 milioni di celle). Per entrambi gli
  estimatori, un fit che supera i limiti dei buffer GPU ripiega sulla CPU con un avviso con `"auto"` e solleva
  `MemoryError` con `"vulkan"` esplicito. Altri buffer del contesto e di lavoro, inclusa la cache di regressione
  a due membri, possono imporre un limite inferiore; 8,4 milioni di celle è un limite superiore. Dettagli in
  [VULKAN.md](VULKAN.md).

In `fp32`, tutti i dispositivi calcolano la stessa funzione: le probabilità coincidono entro circa 3e-5.

`clf.to("cpu")` o `clf.to("cuda[:i]")` sposta un classificatore PyTorch già fitted e i suoi contesti in cache, e
restituisce il classificatore (`clf.to("cpu")` converte anche eventuali tensori fp16 in cache a float32 e
imposta `precision_="fp32"`). Il parametro `device` del costruttore resta invariato; il prossimo `fit` risolve
di nuovo quel parametro. Un classificatore inizializzato con Vulkan non può essere spostato con `to()`; va
rifatto il fit sul dispositivo desiderato.
Anche `reg.to()` sposta solo tra dispositivi torch; per usare Vulkan va rifatto il fit con `device="vulkan"`.

## Precisione

Entrambi gli estimatori accettano `precision="auto"`, `"fp32"` o `"fp16"`:

- `precision="auto"` (predefinito): seleziona l'autocast `"fp16"` su GPU CUDA/ROCm compatibili quando un membro
  ha almeno `FP16_MIN_CELLS` (2.048) celle di contesto (`n_righe_contesto * n_feature >= 2048`), e `"fp32"` su
  tabelle più piccole, su CPU e su Vulkan.
- `precision="fp32"`: forza float32 su qualsiasi dispositivo.
- `precision="fp16"`: richiede un dispositivo CUDA o ROCm (`ValueError` su CPU o Vulkan).
- I decoder a recupero e a momenti restano in float32, e le statistiche del target e l'integrazione dei quantili
  in regressione usano float64.
- La precisione ridotta può cambiare leggermente probabilità, classi predette e quantili rispetto a fp32.
- Se un gruppo del contesto o un blocco di predizione produce valori non finiti in fp16, viene rieseguito
  automaticamente in float32, scrive un messaggio `DEBUG` sul logger `"lightpfn"` e incrementa
  `n_precision_fallbacks_`.
- `precision_` registra la modalità scelta durante il `fit`; cambiare `precision` con `set_params` ha effetto al
  `fit` successivo.

Nella 1.5.1 l’MLP che scala l’attenzione usa GEMM 2D durante l’inferenza GPU a precisione ridotta. Evita
così la chiamata cuBLAS fp16 a batch con stride segnalata in errore con torch 2.15/CUDA 13.2 e driver 580.178
([segnalazione del maintainer](https://github.com/autogluon/tabarena/pull/719#issuecomment-6770068521)). Le
eccezioni CUDA vengono propagate: il ripiego per valori non finiti non recupera un accesso illegale alla
memoria. Dopo un errore CUDA va riavviato il processo prima di provare `precision="fp32"` o torch 2.14.1/CUDA
13.0. Le librerie di compatibilità del container possono caricare una `libcuda` diversa dal driver host
mostrato da `nvidia-smi`; nella diagnosi vanno registrate entrambe. L'errore originale non si è riprodotto su
una RTX 5090 né su una RTX PRO 6000 con lo stesso driver (580.178.04, `libcuda` nativa e di compatibilità) e
torch 2.15/CUDA 13.2, né con il codice 1.5.0 né con il 1.5.1, sui 30 split di anneal della segnalazione.

## Velocità

I tempi misurati di fit più predict sono in [RISULTATI.md](RISULTATI.md#costo). In pratica:

- Una GPU è molto più veloce: su una RX 7900 XT, Vulkan e ROCm sono da 6 a 10 volte più veloci di una CPU
  desktop a 16 thread per la 1.1. Con il protocollo di TabArena (otto modelli di fold senza cache, poi un refit
  in cache), la 1.5.1 addestra 1,65 volte e predice 1,64 volte più veloce della 1.1.0 su una RTX 5090 (medie
  geometriche su 31 task di classificazione; 1,40 e 1,25 volte sulle 17 tabelle fino a 3.000 righe). Le tabelle
  piccole sono limitate dai lanci dei kernel, non dal lavoro della GPU.
- Per bagging o validazione incrociata, crea i modelli dei fold con `cache_context=False`: ognuno predice le
  sue righe di validazione in un solo passaggio.
- Su CPU, `n_estimators=1` è circa quattro volte più veloce di 4 in classificazione (e due volte più veloce di 2
  in regressione) e perde poca accuratezza.
- Imposta `n_threads` al numero di core fisici; hyper-threading e core a basso consumo aiutano poco.
- Predici a blocchi grandi: una chiamata con molte righe è più veloce di molte chiamate con poche righe.

## Pesi, uso offline e checkpoint propri

<a id="pesi-locali-e-migrazione"></a>

Senza `checkpoint` né `model`, il primo `fit` scarica `config.json` e `model.safetensors` dalla repository
Hugging Face `ueuegio/LightPFN`, al commit fissato in `lightpfn/pretrained.json`, tramite `huggingface_hub` e la
sua cache (`HF_HOME`, `HF_HUB_CACHE`). Nella 1.5 la repository contiene le sottocartelle separate `classifier/`
e `regressor/`, e `fit` scarica solo la sottocartella necessaria all'estimatore. Il download non esegue mai
codice della repository.

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor, load_pretrained, save_model

# offline: i pesi devono essere già nella cache di Hugging Face
clf = LightPFNClassifier(local_files_only=True)
reg = LightPFNRegressor(local_files_only=True)

# salva una copia locale di ciascun checkpoint
save_model(load_pretrained(task="classifier"), "weights/classifier")   # scrive model.safetensors e config.json
save_model(load_pretrained(task="regressor"), "weights/regressor")

clf = LightPFNClassifier(checkpoint="weights/classifier", device="cpu", precision="fp32")
reg = LightPFNRegressor(checkpoint="weights/regressor", device="cpu", precision="fp32")

# carica i pesi di classificazione della 1.1 nella 1.5 (file alla radice della revisione 1.1)
old = LightPFNClassifier(revision="bd389ab59a89dd0e05c9ecb7c642c08ee52e9637", precision="fp32")
```

Impostare la variabile d'ambiente `LIGHTPFN_WEIGHTS_DIR=weights` fa leggere a `load_pretrained` direttamente le
sottocartelle locali `weights/classifier/` e `weights/regressor/` senza accesso alla rete.

`load_model(path)` legge i safetensors con la loro configurazione JSON, oppure file PyTorch `.pt` tramite
`torch.load(weights_only=True)`. I checkpoint che richiedono oggetti pickle arbitrari vengono rifiutati di
proposito.

Una cartella `classifier/` o `regressor/` assente a una revisione moderna solleva l’errore di download o
cache, anche offline con una cache incompleta. Il loader non la sostituisce con il vecchio classificatore
nella radice. Le revisioni storiche 1.0/1.1 richieste esplicitamente conservano il formato nella radice.

## Riproducibilità

Con lo stesso `random_state`, gli stessi dati, lo stesso dispositivo, la stessa `precision` e le stesse versioni
delle librerie, `fit`, `predict_proba`, `predict` e `predict_quantiles` restituiscono gli stessi risultati. Con
`precision="fp32"`, tra dispositivi diversi e tra percorso a batch e sequenziale, le predizioni coincidono a
meno degli arrotondamenti in virgola mobile. Con `precision="auto"` su GPU CUDA o ROCm, sopra 2.048 celle di
contesto viene usato fp16 e le predizioni possono differire leggermente da fp32; passa `precision="fp32"` per
confronti con esecuzioni su CPU o con la 1.1. Cambiare l'ordine delle righe di test o la dimensione dei blocchi
di predizione può cambiare gli output di circa 1e-7 in fp32.

## API di basso livello

```python
import torch
from lightpfn import load_pretrained

model = load_pretrained(task="classifier").eval()     # un torch.nn.Module (LightPFN)
net = model.folded()                                  # la stessa funzione con le normalizzazioni incorporate
X_train = torch.randn(1, 500, 8); y_train = torch.randint(0, 2, (1, 500)); X_test = torch.randn(1, 100, 8)
with torch.inference_mode():
    ctx = net.encode(X_train, y_train, n_classes=2)   # batch di problemi: (B, righe, feature)
    logits = net.predict_logits(ctx, X_test)          # (B, righe di test, classi)
```

`lightpfn.Config` contiene l'architettura; `LightPFN(Config())` costruisce la rete di default (non addestrata).
Maggiori dettagli sono nel [report tecnico di LightPFN 1.5](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
e nel [report della 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf).
