import pickle
import numpy as np
from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

menu = [{"name": "KNN", "url": "p_knn"},
        {"name": "Логистическая регрессия", "url": "p_logistic_regression"},
        {"name": "Дерево решений", "url": "p_decision_tree"},
        {"name": "Линейная регрессия", "url": "p_linear_regression"},
        ]


def classification_model_metrics(model: str) -> dict:
    models = {"knn": KNeighborsClassifier(), "logistic_regression": LogisticRegression(),
              "tree": DecisionTreeClassifier(criterion='entropy')}
    students_df = pd.read_excel('model/dataset_students.xlsx')
    students_df.drop_duplicates(inplace=True)
    label_encoder = LabelEncoder()
    students_df["Экзамен"] = label_encoder.fit_transform(students_df["Экзамен"])
    x = students_df.drop(["Экзамен"], axis=1)
    y = students_df["Экзамен"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)
    model = models[model]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def make_df(elem1, elem2, elem3) -> pd.DataFrame:
    return pd.DataFrame({"Средний балл": [float(elem1)],
                         "Посещено лекций": [float(elem2), ],
                         "Выполнено ДЗ": [float(elem3)]})


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k-ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        metrics = classification_model_metrics("knn")
        loaded_model_knn = pickle.load(open('model/knn.bin', 'rb'))
        df = make_df(request.form['list1'], request.form['list2'], request.form['list3'])
        pred = loaded_model_knn.predict(df)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Экзамен сдан" if pred[0] else "Экзамен не сдан",
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])


@app.route("/p_logistic_regression", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        metrics = classification_model_metrics('logistic_regression')
        logistic_regression = pickle.load(open('model/logistic_regression.bin', 'rb'))
        df = make_df(request.form['list1'], request.form['list2'], request.form['list3'])
        pred = logistic_regression.predict(df)
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Экзамен сдан" if pred[0] else "Экзамен не сдан",
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])


@app.route("/p_decision_tree", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        metrics = classification_model_metrics('tree')
        logistic_regression = pickle.load(open('model/tree.bin', 'rb'))
        df = make_df(request.form['list1'], request.form['list2'], request.form['list3'])
        pred = logistic_regression.predict(df)
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Экзамен сдан" if pred[0] else "Экзамен не сдан",
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'], recall=metrics['recall'])


@app.route("/p_linear_regression")
def f_lab4():
    return render_template('lab3.html', title="Линейная регрессия", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)
