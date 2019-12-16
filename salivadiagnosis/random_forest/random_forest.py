from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd


def train(individual_values, dataset):
    # Selecionando as colunas que serão utilizadas no treinamento
    selected_tuples = filter(lambda i: i[1], individual_values)
    selected_cols = [t[0] for t in selected_tuples]
    new_dataset = dataset.filter(selected_cols + ['diagnostico_oral'])

    # Traduzindo os valores dos "labels" dos atributos nominais para números
    le = preprocessing.LabelEncoder()
    trans_dataset_str = new_dataset.select_dtypes(include=['object']).astype(str).apply(le.fit_transform)
    trans_dataset_numeric = new_dataset.select_dtypes(exclude=['object'])
    trans_dataset = pd.concat([trans_dataset_numeric, trans_dataset_str], axis=1)
    x, y = trans_dataset.iloc[:, 0:-1].values.tolist(), trans_dataset.iloc[:, -1].values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Classificando usando "Random Forest"
    clf = RandomForestClassifier(n_estimators=101, n_jobs=-1)
    clf.fit(x_train, y_train)

    return clf.score(x_test, y_test)


def train_pso(individual_values, dataset):
    # Selecionando as colunas que serão utilizadas no treinamento
    selected_cols = [i for i, e in enumerate(individual_values) if e == 1]
    selected_cols.append(-1)
    new_dataset = dataset.iloc[:, selected_cols]

    # Traduzindo os valores dos "labels" dos atributos nominais para números
    le = preprocessing.LabelEncoder()
    trans_dataset_str = new_dataset.select_dtypes(include=['object']).astype(str).apply(le.fit_transform)
    trans_dataset_numeric = new_dataset.select_dtypes(exclude=['object'])
    trans_dataset = pd.concat([trans_dataset_numeric, trans_dataset_str], axis=1)
    x, y = trans_dataset.iloc[:, 0:-1].values.tolist(), trans_dataset.iloc[:, -1].values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # Classificando usando "Random Forest"
    clf = RandomForestClassifier(n_estimators=101, n_jobs=-1)
    clf.fit(x_train, y_train)

    return clf.score(x_test, y_test)
