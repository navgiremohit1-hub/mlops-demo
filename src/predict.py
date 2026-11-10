import joblib

model = joblib.load("../models/iris_model.pkl")
sample_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_data)
print("Predicted class:", prediction)
