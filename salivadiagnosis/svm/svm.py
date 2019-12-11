from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
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
    trans_dataset = pd.concat([trans_dataset_numeric, trans_dataset_str], axis=1)

    # Treinamento da SVM
    classifier = LinearSVC(tol=1e-5)
    inputs, outputs = trans_dataset.iloc[:, 0:-1].values.tolist(), trans_dataset.iloc[:, -1].values.tolist()
    scores = cross_val_score(classifier, inputs, outputs, cv=10)

    return scores.mean()
