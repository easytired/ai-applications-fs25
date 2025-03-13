import gradio as gr
import pandas as pd
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

# Load the model
with open("apartment_price_model.pkl", mode="rb") as f:
    model = pickle.load(f)

# Define Zurich neighborhoods with their approximate coordinates and distances
zurich_neighborhoods = {
    "City Center (Altstadt)": {"lat": 47.3769, "lon": 8.5417, "distance": 0.0},
    "Oerlikon": {"lat": 47.4111, "lon": 8.5458, "distance": 3.8},
    "Altstetten": {"lat": 47.3908, "lon": 8.4889, "distance": 4.2},
    "Wiedikon": {"lat": 47.3708, "lon": 8.5128, "distance": 2.3},
    "Seefeld": {"lat": 47.3550, "lon": 8.5550, "distance": 2.7},
    "Schwamendingen": {"lat": 47.4053, "lon": 8.5648, "distance": 3.5},
    "Wollishofen": {"lat": 47.3517, "lon": 8.5304, "distance": 3.0},
    "Enge": {"lat": 47.3656, "lon": 8.5267, "distance": 1.2},
    "Fluntern": {"lat": 47.3797, "lon": 8.5611, "distance": 1.8},
    "Hottingen": {"lat": 47.3683, "lon": 8.5584, "distance": 1.5},
    "Custom Location": {"lat": 47.3769, "lon": 8.5417, "distance": 0.0}
}

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

def predict_price(neighborhood, rooms, area, has_balcony, is_renovated, custom_lat=None, custom_lon=None):
    # Get coordinates based on neighborhood selection
    if neighborhood == "Custom Location" and custom_lat is not None and custom_lon is not None:
        lat = custom_lat
        lon = custom_lon
    else:
        lat = zurich_neighborhoods[neighborhood]["lat"]
        lon = zurich_neighborhoods[neighborhood]["lon"]
    
    # Calculate distance to city center
    zurich_center_lat = zurich_neighborhoods["City Center (Altstadt)"]["lat"]
    zurich_center_lon = zurich_neighborhoods["City Center (Altstadt)"]["lon"]
    
    if neighborhood == "Custom Location" and custom_lat is not None and custom_lon is not None:
        distance_to_center = haversine_distance(custom_lat, custom_lon, zurich_center_lat, zurich_center_lon)
    else:
        distance_to_center = zurich_neighborhoods[neighborhood]["distance"]
    
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
    result += f"\n- Location: {neighborhood}"
    result += f"\n- {rooms} rooms, {area} m²"
    result += f"\n- {distance_to_center:.2f} km from city center"
    result += f"\n- {'Has balcony' if has_balcony else 'No balcony'}"
    result += f"\n- {'Renovated' if is_renovated else 'Not renovated'}"
    
    return result

# Function to update visibility of lat/lon inputs based on neighborhood selection
def update_custom_location(neighborhood):
    if neighborhood == "Custom Location":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

# Function to reset all inputs to default values
def reset_inputs():
    return [
        "City Center (Altstadt)",  # neighborhood
        47.3769,                   # custom_lat
        8.5417,                    # custom_lon
        3.5,                       # rooms
        75,                        # area
        True,                      # has_balcony
        False,                     # is_renovated
        ""                         # clear output
    ]

# Create Gradio interface with neighborhood dropdown
with gr.Blocks() as demo:
    gr.Markdown("# Zurich Apartment Rent Prediction")
    gr.Markdown("""
    This app predicts apartment rental prices in Zurich with a special feature: Distance to City Center.
    
    **Special Feature Description:**
    The model automatically calculates how far the apartment is from Zurich city center.
    This distance is a critical factor in real estate pricing - properties closer to the city center typically
    command higher rents due to convenience and accessibility to urban amenities.
    
    Simply select a neighborhood, and the app will use its distance from the city center to help
    provide a more accurate rental price prediction.
    """)
    
    with gr.Row():
        with gr.Column():
            neighborhood = gr.Dropdown(
                label="Neighborhood", 
                choices=list(zurich_neighborhoods.keys()),
                value="City Center (Altstadt)"
            )
            custom_lat = gr.Number(label="Custom Latitude", value=47.3769, visible=False)
            custom_lon = gr.Number(label="Custom Longitude", value=8.5417, visible=False)
            rooms = gr.Number(label="Number of Rooms", value=3.5)
            area = gr.Number(label="Area (m²)", value=75)
            has_balcony = gr.Checkbox(label="Has Balcony", value=True)
            is_renovated = gr.Checkbox(label="Is Renovated", value=False)
            
            with gr.Row():
                submit_button = gr.Button("Submit")
                clear_button = gr.Button("Clear")
        
        with gr.Column():
            output = gr.Textbox(label="Output")
    
    # Connect the neighborhood dropdown to show/hide custom lat/lon
    neighborhood.change(
        fn=update_custom_location,
        inputs=neighborhood,
        outputs=[custom_lat, custom_lon]
    )
    
    # Connect the submit button
    submit_button.click(
        fn=predict_price,
        inputs=[neighborhood, rooms, area, has_balcony, is_renovated, custom_lat, custom_lon],
        outputs=output
    )
    
    # Connect the clear button
    clear_button.click(
        fn=reset_inputs,
        inputs=None,
        outputs=[neighborhood, custom_lat, custom_lon, rooms, area, has_balcony, is_renovated, output]
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["Oerlikon", 3.5, 75, True, True],
            ["Seefeld", 2.0, 60, False, False],
            ["Wiedikon", 4.5, 120, True, False],
        ],
        inputs=[neighborhood, rooms, area, has_balcony, is_renovated],
        outputs=output,
        fn=predict_price
    )

demo.launch()