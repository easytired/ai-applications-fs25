# %%
import gradio as gr
import pandas as pd
from sklearn.datasets import load_iris
import pickle

# Load model from file
model_filename = "iris_random_forest_classifier.pkl" # hier dann file name mit neuem pkl ersetzen
with open(model_filename, mode="rb") as f:
    model = pickle.load(f) # mein model das lädt also von pkl
# Load dataset
iris = load_iris(as_frame=True)  # ladate daten sprich appartment daten

def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=iris.feature_names)
    prediction = model.predict(input_data)[0]
    return iris.target_names[prediction]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width"),
    ],
    outputs="text",
    examples=[
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [7.7, 3.8, 6.7, 2.2],
    ],
    title="Iris Flower Prediction",
    description="Enter the sepal and petal measurements to predict the Iris species."
)

demo.launch()
# %%
# Load model from file
model_filename = "flat_price_predictor.pkl"
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

def predict(postcode, area, number_of_rooms):
    input_data = pd.DataFrame([[postcode, area, number_of_rooms]],
                              columns=["postcode", "area", "number_of_rooms"])
    prediction = model.predict(input_data)[0]
    return f"The predicted price of the flat is: ${prediction:.2f}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Postcode"),
        gr.Number(label="Area (in square meters)"),
        gr.Number(label="Number of Rooms"),
    ],
    outputs="text",
    title="Flat Price Prediction",
    description="Enter the postcode, area, and number of rooms to predict the price of a flat."
)

demo.launch()