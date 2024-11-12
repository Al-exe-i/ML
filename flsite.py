import os
import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify, redirect
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from model.neuro import SingleNeuron
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# menu = [{"name": "KNN", "url": "p_knn"},
#         {"name": "Логистическая регрессия", "url": "p_logistic_regression"},
#         {"name": "Дерево решений", "url": "p_decision_tree"},
#         {"name": "Линейная регрессия", "url": "p_linear_regression"},
#         {"name": "Нейронка", "url": "p_neural_network"},
#         {"name": "Нейронка для одежды", "url": 'clothes_neural'},
#         {"name": "Классификация телефонов", "url": 'phones_cnn'},
#         ]

menu = [
        {"name": "Нейронка", "url": "p_neural_network"},
        {"name": "Нейронка для одежды", "url": 'clothes_neural'},
        {"name": "Классификация телефонов", "url": 'phones_cnn'},
        ]

new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('model/neuron_weights.txt')
model = tf.keras.models.load_model('model/regression.h5')
clothes_model = tf.keras.models.load_model('model/clothes_model.h5')
cnn_phones_model = tf.keras.models.load_model('model/phones_cnn.h5')


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
    return {"accuracy": round(accuracy, 3), "precision": round(precision, 3), "recall": round(recall, 3)}


def make_df(elem1, elem2, elem3) -> pd.DataFrame:
    return pd.DataFrame({"Средний балл": [float(elem1)],
                         "Посещено лекций": [float(elem2), ],
                         "Выполнено ДЗ": [float(elem3)]})


@app.route("/")
def index():
    return render_template('index.html', title="Машинное обучение. Мишин Алексей ИСТ-301", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_knn():
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
def f_log_regression():
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
def f_decision_tree():
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


@app.route("/p_linear_regression", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        ###
        model = LinearRegression()
        dataset = pd.read_excel('model/Dataset4.xlsx')
        dataset['Пол'] = LabelEncoder().fit_transform(dataset['Пол'])
        x = dataset[['Рост', 'Вес', 'Пол']]
        y = dataset['Размер обуви']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r_2 = r2_score(y_test, y_pred)
        ###
        linear_regression = pickle.load(open('model/linear.bin', 'rb'))
        pred = linear_regression.predict([[float(request.form['list1']), float(request.form['list2']),
                                           float(request.form['list3'])]])
        return render_template('lab4.html', title="Линейная регрессия", menu=menu,
                               class_model=round(pred[0]),
                               mae=mae,
                               mse=mse, r_2=r_2)


@app.route("/p_neural_network", methods=["GET", "POST"])
def p_neural_network():
    if request.method == "GET":
        return render_template('lab14.html', title="Нейрон", menu=menu, class_model='')
    if request.method == "POST":
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'п', 'л'))
        return render_template('lab14.html', title="Нейрон", menu=menu,
                               class_model="Это: " + str(
                                   *np.where(predictions >= 0.5, 'практическая аудитория', 'лекционная аудитория')))


@app.route("/api", methods=['GET', 'POST'])
def api():
    try:
        request_data = request.get_json()
        model_type = request_data['model']
    except ValueError as e:
        return jsonify(status='error', message=str(e))
    except Exception as e:
        return jsonify(status='Unknown error', message=str(e))
    if model_type == 'linear_regression':
        try:
            height = request_data['height']
            weight = request_data['weight']
            gender = request_data['gender']
            linear_regression = pickle.load(open('model/linear.bin', 'rb'))
            pred = linear_regression.predict([[float(height), float(weight), float(gender)]])
            return jsonify(size=round(pred[0]))
        except ValueError as e:
            return jsonify(status='error', message=str(e))
        except Exception as e:
            return jsonify(status='Unknown error', message=str(e))
    else:
        try:
            av_score = request_data['score']
            lectures = request_data['lectures']
            homeworks = request_data['homeworks']
        except ValueError as e:
            return jsonify(status='error', message=str(e))
        except Exception as e:
            return jsonify(status='unknown error', message=str(e))
        try:
            if model_type == 'knn':
                loaded_model_knn = pickle.load(open('model/knn.bin', 'rb'))
                df = make_df(av_score, lectures, homeworks)
                pred = loaded_model_knn.predict(df)
                return jsonify(result="Экзамен сдан" if pred[0] else "Экзамен не сдан")
            elif model_type == 'logistic_regression':
                loaded_model_knn = pickle.load(open('model/logistic_regression.bin', 'rb'))
                df = make_df(av_score, lectures, homeworks)
                pred = loaded_model_knn.predict(df)
                return jsonify(result="Экзамен сдан" if pred[0] else "Экзамен не сдан")
            elif model_type == 'decision_tree':
                loaded_model_knn = pickle.load(open('model/knn.bin', 'rb'))
                df = make_df(av_score, lectures, homeworks)
                pred = loaded_model_knn.predict(df)
                return jsonify(result="Экзамен сдан" if pred[0] else "Экзамен не сдан")
        except ValueError as e:
            return jsonify(status='error', message=str(e))
        except Exception as e:
            return jsonify(status="unknown error", message=str(e))


@app.route('/api/v1')
def api_v1():
    if request.method == 'GET':
        height = float(request.args.get('height'))
        weight = float(request.args.get('weight'))
        gender = float(request.args.get('gender'))
        result = model.predict(np.array([[height, weight, gender]]))[0][0]
        return jsonify(size=round(result))


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')  # Загружаем изображение
    img_array = image.img_to_array(img)  # Преобразуем изображение в массив
    img_array = 255 - img_array  # Нормализуем
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность
    return img_array


@app.route('/clothes_neural', methods=['GET', 'POST'])
def clothes_neural():
    if request.method == 'GET':
        return render_template('lab17.html', menu=menu, title="Нейронная сеть", post_url=url_for('clothes_neural'))
    elif request.method == 'POST':
        file = request.files['image']
        if file:
            class_names = [
                'Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
                'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки'
            ]
            filename = file.filename
            file.save(f'{filename}')
            prepared_image = load_and_preprocess_image(f'{filename}')
            predictions = clothes_model.predict(prepared_image)
            predicted_class = class_names[predictions[0].argmax()]
            os.remove(filename)
            return render_template('lab17.html', predicted_class=predicted_class, menu=menu, title="Нейронная сеть")
        else:
            return redirect(url_for('clothes_neural'))


@app.route('/phones_cnn', methods=['Get', 'POST'])
def phones_cnn():
    if request.method == 'GET':
        return render_template('lab19.html', menu=menu, title="Нейронная сеть", post_url=url_for('phones_cnn'))
    elif request.method == 'POST':
        file = request.files['image']
        img_height = 180
        img_width = 180
        if file:
            class_names = ['кнопочный телефон', 'смартфон']
            filename = file.filename
            file.save(f'{filename}')
            img = tf.keras.utils.load_img(filename, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch
            predictions = cnn_phones_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            os.remove(filename)
            return render_template('lab19.html', predicted_class=predicted_class, menu=menu, title="Нейронная сеть",
                                   percent=round(np.max(score) * 100, 1))
        else:
            return redirect(url_for('phones_cnn'))


if __name__ == "__main__":
    app.run(debug=True)
