import gradio as gr
import pandas as pd
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

# Load only the model
with open("apartment_price_model.pkl", mode="rb") as f:
    model = pickle.load(f)

# Define features directly in the app
features = [
    'rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income',
    'price_per_room', 'distance_to_center', 'has_balcony', 'is_renovated'
]

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

# Zurich city center coordinates
zurich_center_lat = 47.3769
zurich_center_lon = 8.5417

def predict_price(rooms, area, population, pop_density, foreign_pct, 
                 employment, tax_income, lat, lon, has_balcony, is_renovated):
    
    # Calculate derived features
    price_per_room = 0  # This will be estimated by the model
    distance_to_center = haversine_distance(lat, lon, zurich_center_lat, zurich_center_lon)
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        'rooms': rooms,
        'area': area,
        'pop': population,
        'pop_dens': pop_density,
        'frg_pct': foreign_pct,
        'emp': employment,
        'tax_income': tax_income,
        'price_per_room': price_per_room,
        'distance_to_center': distance_to_center,
        'has_balcony': has_balcony,
        'is_renovated': is_renovated
    }])
    
    # Make prediction
    predicted_price = model.predict(input_data[features])[0]
    
    # Update price_per_room with the predicted price
    price_per_room = predicted_price / rooms
    input_data['price_per_room'] = price_per_room
    
    # Make another prediction with updated price_per_room
    predicted_price = model.predict(input_data[features])[0]
    
    # Format the result
    result = f"Predicted Monthly Rent: CHF {predicted_price:.0f}"
    
    # Additional insights
    result += f"\n\nProperty Details:"
    result += f"\n- {rooms} rooms, {area} m²"
    result += f"\n- {distance_to_center:.2f} km from city center"
    result += f"\n- {'Has balcony' if has_balcony else 'No balcony'}"
    result += f"\n- {'Renovated' if is_renovated else 'Not mentioned as renovated'}"
    
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Number of Rooms"),
        gr.Number(label="Area (m²)"),
        gr.Number(label="Population", value=420217),
        gr.Number(label="Population Density", value=4778),
        gr.Number(label="Foreign Percentage", value=32.45),
        gr.Number(label="Employment", value=491193),
        gr.Number(label="Tax Income", value=85446),
        gr.Number(label="Latitude", value=47.3769),
        gr.Number(label="Longitude", value=8.5417),
        gr.Checkbox(label="Has Balcony"),
        gr.Checkbox(label="Is Renovated"),
    ],
    outputs="text",
    examples=[
        [3.5, 75, 420217, 4778, 32.45, 491193, 85446, 47.41106, 8.54654, True, True],
        [2.0, 60, 420217, 4778, 32.45, 491193, 85446, 47.37624, 8.52814, False, False],
        [4.5, 120, 420217, 4778, 32.45, 491193, 85446, 47.36368, 8.54678, True, False],
    ],
    title="Zurich Apartment Rent Prediction",
    description="Enter apartment details to predict the monthly rent in Swiss Francs (CHF)."
)

demo.launch()