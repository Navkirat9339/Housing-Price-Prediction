from predict import load_model, predict_house_value

model = load_model()

example_house = {
    "MedInc": 5.0,
    "HouseAge": 30.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}

prediction = predict_house_value(model, example_house)

print(f"Predicted target: {prediction:.3f}")
print(f"Predicted value: ${prediction * 100_000:,.0f}")