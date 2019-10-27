import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Glucose':89, 'Insulin':94, 'BMI':28.1, 'Age':21})

print(r.json())
