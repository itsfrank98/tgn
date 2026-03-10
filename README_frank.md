## Per addestrare un nuovo modello:
1. Lancia `preprocess_data.py`
2. Lancia `train_self_supervised.py`
   3. All'interno di `get_data_with_interaction` potresti dover modificare i numeri degli snapshot di validation e test

## I campi del parameters.yaml da cambiare per i vari test sono:
  - `data: "gab_with_synthetic"`: Source da dove prendere i dati preprocessati
  - `consider_synthetic: True`: Se considerare gli utenti sintetici
  - `load_model: True`: Se caricare un modello pre addestrato
  - `negative_sampling_ratio: 1`

## Dopo aver addestrato:
- Lancia `train_self_supervised.py` settando `load_model: True` nel file parameters.yaml
- Sarà generato un file `connections_n.pkl` dove `n` è il negative_sampling_rtio