import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_individual_travels_final(travel_data):
    """
    Adapted version to dynamically handle any parking location types and plot the daily travels of an individual,
    including true initial and final parking information with dynamic color coding.
    
    Parameters:
    - travel_data: DataFrame containing the travel information for an individual, including departure and arrival times,
      parking locations, departure locations, trip number, and parking information.
    """
    # Ensure the data is sorted by departure time
    travel_data_sorted = travel_data.sort_values(by='Heure_depart')
    
    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Dynamically create a color map for parking locations
    parking_locations = pd.concat([travel_data_sorted['Lieu_depart'], travel_data_sorted['Lieu_arrivee']]).unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(parking_locations)))
    color_map = {location: color for location, color in zip(parking_locations, colors)}
    
    # Plot initial parking segment (from midnight to first departure)
    first_departure = travel_data_sorted.iloc[0]['Heure_depart']
    initial_parking_color = color_map[travel_data_sorted.iloc[0]['Lieu_depart']]
    ax.plot([0, first_departure], [1, 1], color=initial_parking_color, linewidth=8)
    
    # Loop through each trip to plot
    for index, row in travel_data_sorted.iterrows():
        start = row['Heure_depart']
        end = row['Heure_arrivee']
        parking_location = row['Lieu_arrivee']
        
        # Plot the travel segment
        ax.plot([start, end], [1, 1], color='black', linewidth=8)  # Uniform line thickness for travel
        
        # Plot the parking segment with slight spacing
        if index < len(travel_data_sorted) - 1:
            next_start = travel_data_sorted.iloc[index + 1]['Heure_depart']
            parking_color = color_map[parking_location]
            ax.plot([end, next_start], [1, 1], color=parking_color, linewidth=8)
    
    # Plot final parking segment (from last arrival to midnight)
    last_arrival = travel_data_sorted.iloc[-1]['Heure_arrivee']
    final_parking_color = color_map[travel_data_sorted.iloc[-1]['Lieu_arrivee']]
    ax.plot([last_arrival, 24], [1, 1], color=final_parking_color, linewidth=8)
    
    # Improving the plot aesthetics
    ax.set_xlim(0, 24)  # Set x-axis to span from midnight to midnight
    ax.set_yticks([])  # Hide y-axis as it's not relevant
    ax.set_xlabel("Heure")
    plt.title("Déplacements journaliers d'un individu avec stationnements dynamiques")
    # Create legend entries for parking locations
    legend_entries = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    # Add legend to the plot
    ax.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

