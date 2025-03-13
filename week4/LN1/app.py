import gradio as gr
import pandas as pd
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

# Load the model
with open("apartment_price_model.pkl", mode="rb") as f:
    model = pickle.load(f)

# Zurich city center coordinates
zurich_center_lat = 47.3769
zurich_center_lon = 8.5417

# Function to calculate distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def predict_price(rooms, area, lat, lon, has_balcony, is_renovated):
    # Calculate special feature: distance to city center
    distance_to_center = haversine_distance(lat, lon, zurich_center_lat, zurich_center_lon)
    
    # Default values for other features
    pop = 420217
    pop_dens = 4778
    frg_pct = 32.45
    emp = 491193
    tax_income = 85446
    price_per_room = 0  # This will be updated by the model
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        'rooms': rooms,
        'area': area,
        'pop': pop,
        'pop_dens': pop_dens,
        'frg_pct': frg_pct,
        'emp': emp,
        'tax_income': tax_income,
        'price_per_room': price_per_room,
        'distance_to_center': distance_to_center,
        'has_balcony': 1 if has_balcony else 0,
        'is_renovated': 1 if is_renovated else 0
    }])
    
    # Define features in the correct order
    features = [
        'rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income',
        'price_per_room', 'distance_to_center', 'has_balcony', 'is_renovated'
    ]
    
    # Make prediction
    predicted_price = model.predict(input_data[features])[0]
    
    # Format the result
    result = f"Predicted Monthly Rent: CHF {predicted_price:.0f}"
    result += f"\n\nProperty Details:"
    result += f"\n- {rooms} rooms, {area} m²"
    result += f"\n- {distance_to_center:.2f} km from city center"
    result += f"\n- {'Has balcony' if has_balcony else 'No balcony'}"
    result += f"\n- {'Renovated' if is_renovated else 'Not renovated'}"
    
    return result

# Create Gradio interface with fewer inputs
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Number of Rooms"),
        gr.Number(label="Area (m²)"),
        gr.Number(label="Latitude", value=47.3769),
        gr.Number(label="Longitude", value=8.5417),
        gr.Checkbox(label="Has Balcony"),
        gr.Checkbox(label="Is Renovated"),
    ],
    outputs="text",
    examples=[
        [3.5, 75, 47.41106, 8.54654, True, True],
        [2.0, 60, 47.37624, 8.52814, False, False],
        [4.5, 120, 47.36368, 8.54678, True, False],
    ],
    title="Zurich Apartment Rent Prediction",
    description="""
    This app predicts apartment rental prices in Zurich with a special feature: Distance to City Center.
    
    **Special Feature Description:**
    The app automatically calculates the apartment's distance from Zurich city center using the Haversine formula.
    This distance is a critical factor in real estate pricing - properties closer to the city center typically
    command higher rents due to convenience and accessibility to urban amenities.
    
    Simply enter the apartment's latitude and longitude, and the model will incorporate this distance
    calculation to provide a more accurate rental price prediction.
    """
)

demo.launch()