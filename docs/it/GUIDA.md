# Guida all'uso

[English](../en/GUIDE.md) · **Italiano**

- [Cosa fa un fit](#cosa-fa-un-fit)
- [Parametri](#parametri)
- [Attributi dopo il fit](#attributi-dopo-il-fit)
- [Dati in ingresso](#dati-in-ingresso)
- [Tabelle grandi](#tabelle-grandi)
- [Dispositivi](#dispositivi)
- [Velocità](#velocità)
- [Pesi, uso offline e checkpoint propri](#pesi-uso-offline-e-checkpoint-propri)
- [Riproducibilità](#riproducibilità)
- [API di basso livello](#api-di-basso-livello)

## Cosa fa un fit

`fit(X, y)` non addestra nulla. Valida i dati, carica la rete preaddestrata (una volta per classificatore) e
codifica il training set: statistiche delle colonne, stati dei punti induttori dei due stadi per colonna e
chiavi e valori dei blocchi di in-context learning. `predict_proba(X)` fa passare nella rete solo le righe di
test, a blocchi, e le fa guardare quel contesto in cache. La predizione di una riga di test non dipende mai dalle
altre righe di test.

Questo descrive il default `cache_context=True`. Con `cache_context=False`, fit conserva copie degli input;
ogni predizione costruisce lo stesso contesto e poi lo libera. Le predizioni non cambiano.

Con `n_estimators > 1`, il primo estimatore usa l'ordine originale delle feature e degli slot di classe, ogni
altro una permutazione casuale delle feature e un'assegnazione casuale delle classi agli slot di etichetta del
modello; le probabilità vengono mediate. Su TabArena quattro estimatori alzano l'AUC media di circa 0,13 punti
rispetto a uno e migliorano il rank medio su sette modelli da 3,37 a 2,50, a un costo quattro volte maggiore.

## Parametri

```python
LightPFNClassifier(model=None, checkpoint=None, device="auto", n_estimators=4, max_context=20000,
                   chunk_rows=2048, n_threads=None, seed=0, *, chunk_cells="auto", batch_cells="auto",
                   fold=True, random_state=None, repo_id=None, revision=None, cache_dir=None,
                   local_files_only=False, cache_context=True)
```

| Parametro | Default | Significato |
|---|---|---|
| `n_estimators` | `4` | estimatori mediati su permutazioni di feature e slot di etichetta; 1 è circa quattro volte più veloce su CPU |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda[:i]"`, `"vulkan[:i]"`; vedi [Dispositivi](#dispositivi) |
| `max_context` | `20000` | oltre questo numero di righe di training, ogni estimatore legge un sottoinsieme stratificato di questa dimensione |
| `random_state` | `None` | seme di permutazioni e sottoinsiemi (`seed` è il vecchio alias; impostane uno solo) |
| `n_threads` | `None` | thread CPU di PyTorch (`torch.set_num_threads`, vale per tutto il processo); `None` lascia l'impostazione di PyTorch |
| `chunk_rows` | `2048` | righe di test per blocco di predizione (limita la memoria) |
| `checkpoint` | `None` | un `model.safetensors` locale (con `config.json` accanto), la sua cartella o un file `.pt` |
| `model` | `None` | un modulo torch `LightPFN` da usare al posto dei pesi scaricati (copiato al fit) |
| `repo_id`, `revision` | `None` | un'altra repository Hugging Face; una repository personalizzata richiede una revisione esplicita |
| `cache_dir` | `None` | cartella della cache Hugging Face |
| `local_files_only` | `False` | non usare mai la rete; i pesi devono essere già in cache |
| `cache_context` | `True` | conserva i contesti codificati tra le predizioni; `False` conserva copie degli input e ricostruisce e libera i contesti a ogni predizione, utile per i modelli dei fold usati una volta sola |
| `chunk_cells`, `batch_cells`, `fold` | `"auto"`, `"auto"`, `True` | implementazione dell'inferenza: cache blocking, batch degli estimatori su GPU e rete "piegata". Cambiano velocità e memoria, non le predizioni (a meno degli arrotondamenti). `chunk_cells=None, batch_cells=0, fold=False` è il percorso semplice |

Il costruttore memorizza i parametri e nient'altro, quindi `clone`, `get_params` e `set_params` funzionano come
in qualsiasi estimatore scikit-learn e non scaricano nulla né toccano la GPU.

## Attributi dopo il fit

| Attributo | Contenuto |
|---|---|
| `classes_` | le etichette delle classi, nell'ordine delle colonne di `predict_proba` |
| `n_features_in_`, `feature_names_in_` | numero e nomi (per DataFrame con nomi di colonna stringa) delle feature |
| `is_categorical_` | maschera booleana delle colonne codificate come categorie (input pandas) |
| `categories_` | dizionario dalla posizione della colonna al vocabolario categorico del fit; un dtype category conserva tutti i livelli dichiarati, anche quelli inutilizzati |
| `device_` | il dispositivo del backend selezionato, aggiornato da `to()` |
| `fit_device_` | il dispositivo che esegue davvero l'inferenza, compreso un eventuale ripiego automatico sulla CPU |
| `model_` | la rete `LightPFN` caricata |

## Dati in ingresso

- **Array**: qualsiasi array numerico di forma `(n_righe, n_feature)`. NaN indica un valore mancante; valori
  infiniti e matrici sparse vengono rifiutati.
- **DataFrame pandas**: le colonne di tipo category, stringa, object o bool diventano codici ordinali delle
  categorie viste in `fit` (nell'ordine di pandas: valori ordinati per le stringhe, l'ordine del dtype per un
  dtype category, compresi i livelli dichiarati inutilizzati). I valori mancanti e i valori fuori da quel vocabolario diventano NaN. Le colonne numeriche,
  comprese le nullable `Int64` e `Float64`, diventano float con `pd.NA` come NaN. In predizione le colonne devono
  avere gli stessi nomi e lo stesso ordine del `fit`.
- **Etichette**: qualsiasi etichetta accettata da scikit-learn (interi, stringhe). Il modello è stato addestrato
  su 2-10 classi; oltre 10 viene sollevato un errore. Con una sola classe, `predict_proba` restituisce 1 per
  quella classe.
- **Numero di feature**: il modello è stato addestrato fino a 100 feature e funziona anche con di più; le
  tabelle larghe costano di più (gli stadi sulle celle sono lineari nel numero di feature).
- **Scalatura**: non serve. Ogni colonna viene standardizzata e trasformata in ranghi sulle righe di training,
  dentro il modello.

Le colonne categoriche vengono lette come codici ordinali, quindi il modello ne vede l'ordine. Va bene per
cardinalità bassa e media; per colonne con migliaia di livelli restano più forti i codificatori a statistiche
del target come quello di CatBoost (vedi la roadmap nel README).

## Tabelle grandi

L'attenzione di in-context learning cresce con il quadrato della lunghezza del contesto, quindi tempo e memoria
crescono in fretta con il numero di righe di training.

- Fino a `max_context` (20.000) righe, ogni estimatore legge tutto il training set.
- Oltre, ogni estimatore legge un proprio sottoinsieme stratificato di `max_context` righe che tiene ogni classe
  (fino a cinque righe ciascuna, se conteggi e budget del contesto lo permettono). Più estimatori coprono allora una parte maggiore dei dati.
- `max_context` si può alzare. Il report valuta contesti fino a 100.000 righe, dove il modello è in media alla
  pari con CatBoost di default; il costo su CPU è alto (circa 75 s per 94.000 righe con un estimatore su 24
  thread).

## Dispositivi

`device="auto"` prova nell'ordine: una GPU CUDA o ROCm tramite PyTorch (`torch.cuda.is_available()`), una GPU con
driver Vulkan se `wgpu` è installato, poi la CPU. La variabile d'ambiente `LIGHTPFN_DEVICE` sostituisce `"auto"`
(per esempio `LIGHTPFN_DEVICE=cpu`). Un dispositivo esplicito viene usato così com'è, con un errore se non è
disponibile.

Su Vulkan, le celle di training di un estimatore (righe per feature per 64 float) devono stare in un singolo
buffer della GPU (2 GiB sul driver della RX 7900 XT provato, circa 8,4 milioni di celle). Se non ci stanno, `"auto"`
ripiega sulla CPU con un avviso e un `"vulkan"` esplicito solleva un `MemoryError`. Dettagli in
[VULKAN.md](VULKAN.md).
Altri buffer del contesto e di lavoro possono imporre un limite inferiore; 8,4 milioni di celle è un limite superiore.

Tutti i dispositivi calcolano la stessa funzione: le probabilità coincidono entro circa 3e-5.

`clf.to("cpu")` o `clf.to("cuda[:i]")` sposta un classificatore PyTorch già fitted e i suoi contesti in cache,
e restituisce il classificatore. Il parametro `device` del costruttore resta invariato; il prossimo `fit`
risolve di nuovo quel parametro. Un classificatore inizializzato con Vulkan non può essere spostato con
`to()`; va rifatto il fit sul dispositivo desiderato.

## Velocità

I tempi misurati di fit più predict sono in [RISULTATI.md](RISULTATI.md#costo). In pratica:

- Una GPU è molto più veloce: su una RX 7900 XT, Vulkan e ROCm sono da 6 a 10 volte più veloci di una CPU
  desktop a 16 thread.
- Su CPU, `n_estimators=1` è circa quattro volte più veloce di 4 e perde poca accuratezza.
- Imposta `n_threads` al numero di core fisici; hyper-threading e core a basso consumo aiutano poco.
- Predici a blocchi grandi: una chiamata con molte righe è più veloce di molte chiamate con poche righe.

## Pesi, uso offline e checkpoint propri

Senza `checkpoint` né `model`, il primo `fit` scarica `config.json` e `model.safetensors` dalla repository Hugging
Face `ueuegio/LightPFN`, al commit fissato in `lightpfn/pretrained.json`, tramite `huggingface_hub` e la sua cache
(`HF_HOME`, `HF_HUB_CACHE`). Il download non esegue mai codice della repository.

```python
# offline: i pesi devono essere già in cache
clf = LightPFNClassifier(local_files_only=True)

# una copia locale
from lightpfn import load_pretrained, save_model
save_model(load_pretrained(), "lightpfn_weights/")      # scrive model.safetensors e config.json
clf = LightPFNClassifier(checkpoint="lightpfn_weights/")
```

`load_model(path)` legge i safetensors con la loro configurazione JSON, oppure file PyTorch tramite
`torch.load(weights_only=True)`. I checkpoint che richiedono oggetti pickle arbitrari vengono rifiutati di
proposito.

## Riproducibilità

Con lo stesso `random_state`, gli stessi dati, lo stesso dispositivo e le stesse versioni delle librerie, `fit`
e `predict_proba` restituiscono le stesse probabilità. Tra dispositivi diversi, e tra percorso a batch e
sequenziale, coincidono a meno degli arrotondamenti in virgola mobile. Cambiare l'ordine delle righe di test o
la dimensione dei blocchi di predizione può cambiare le probabilità di circa 1e-7.

## API di basso livello

```python
import torch
from lightpfn import load_pretrained

model = load_pretrained().eval()                      # un torch.nn.Module (LightPFN)
net = model.folded()                                  # la stessa funzione con le normalizzazioni incorporate
X_train = torch.randn(1, 500, 8); y_train = torch.randint(0, 2, (1, 500)); X_test = torch.randn(1, 100, 8)
with torch.inference_mode():
    ctx = net.encode(X_train, y_train, n_classes=2)   # batch di problemi: (B, righe, feature)
    logits = net.predict_logits(ctx, X_test)          # (B, righe di test, classi)
```

`lightpfn.Config` contiene l'architettura; `LightPFN(Config())` costruisce la rete di default (non addestrata).
