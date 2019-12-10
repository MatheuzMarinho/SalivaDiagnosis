from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import pandas as pd


def train(individual, dataset):
    # Selecionando as colunas que serão utilizadas no treinamento
    selected_tuples = filter(lambda x: x[1], individual)
    selected_cols = [t[0] for t in selected_tuples]
    new_dataset = dataset.filter(selected_cols + ['diagnostico_oral'])

    # Traduzindo os valores dos "labels" dos atributos nominais para números
    le = preprocessing.LabelEncoder()
    trans_dataset_str = new_dataset.select_dtypes(include=['object']).astype(str).apply(le.fit_transform)
    trans_dataset_numeric = new_dataset.select_dtypes(exclude=['object'])
    trans_dataset = pd.concat([trans_dataset_str, trans_dataset_numeric], axis=1)

    # Treinando a rede
    inputs, outputs = trans_dataset.iloc[:, 0:-1].values.tolist(), trans_dataset.iloc[:, -1].values.tolist()

    # TODO: separar bases de treino e validação, treinar e validar a rede
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(inputs, outputs)

    return None
