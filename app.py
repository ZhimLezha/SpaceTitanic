import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Загрузка модели
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    input_data = request.json

    # Преобразуем данные в формат, подходящий для модели
    features = [input_data['Age'],
                input_data['RoomService'],
                input_data['FoodCourt'],
                input_data['ShoppingMall'],
                input_data['Spa'],
                input_data['Cabin_2'],
                input_data['VRDeck'],
                input_data['CryoSleep'],
                input_data['VIP'],
                input_data['HomePlanet'],
                input_data['Destination'],
                input_data['Cabin_1'],
                input_data['Cabin_3'],
                input_data['Name_1'],
                input_data['Name_2']            
           ]

    # Выполняем предсказание
    prediction = model.predict([features])

    # Возвращаем результат в формате JSON
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

