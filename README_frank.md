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
- Sarà generato un file `connections_n.pkl` dove `n` è il negative_sampling_ratio


### Come funziona il preprocessing?
Il preprocessing genera un csv con questi campi:
- source_user: ID dell'utente da cui l'arco parte
- destination_user: ID dell'utente a cui l'arco punta
- timestamp
- state_label,
- comma_separated_list_of_features,
- type_interaction: può essere 0 (indica l'evoluzione del nodo stesso), 1 (indica due utenti che si seguono), 2 (indica un commento)