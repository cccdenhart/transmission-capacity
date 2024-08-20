import json
import os

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
import rasterio
from typing import List, Tuple

load_dotenv()

@st.cache_data
def load_data() -> pd.DataFrame:
    transmission_fp = "data/pjm_transmission.geojson"
    with open(transmission_fp) as f:
        gj = json.load(f)

    geo_df = gpd.GeoDataFrame.from_features(gj["features"])

    cost_fp = "data/pjm_july_lmp.csv"
    cost_df = pd.read_csv(cost_fp)

    geo_cost_df = geo_df.merge(cost_df, left_on="SUB_1", right_on="pnode_name", how="inner")
    avg_geo_cost_df = (
        geo_cost_df.groupby(
            ["ID", "pnode_id", "pnode_name", "SUB_1", "SUB_2", "VOLTAGE", "geometry"]
        )
        .mean("congestion_price_rt")
        .reset_index()
    )

    return avg_geo_cost_df

def load_solar_tif() -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    tif = "data/pvout_atlantic.tif"
    src = rasterio.open(tif)
    array = src.read()
    bounds = src.bounds
    bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
    return array, bbox

# Base page info
st.title("Energy Generation Planning App")

layer = st.selectbox("Feature", [None, "Congestion Cost"])

# Load data
df = load_data()

#Load raster data
pv, bbox = load_solar_tif()

# Initialize a Folium map
m = folium.Map(location=[39.653806, -77.152707], zoom_start=7)

# Add LayerControl to the map
m.add_child(folium.LayerControl())

# Add the colormap legend to the map
costs = df["congestion_price_da"]
min_value, max_value = min(costs), max(costs)
colormap = cm.linear.YlOrRd_09.scale(min_value, max_value)
colormap.add_to(m)

#add solar potential raster data to map
img = folium.raster_layers.ImageOverlay(
    name="PV OUT",
    image=np.moveaxis(pv, 0, -1),
    bounds=bbox,
    interactive=True,
    cross_origin=False,
    zindex=1,
    show=False,
)
img.add_to(m)

for _, row in df.iterrows():
    linestrings = row["geometry"]
    name = row["ID"]
    cost = row["congestion_price_da"]

    for linestring in linestrings.geoms:
        locations = np.flip(np.stack(linestring.xy, axis=1), axis=1)
        if layer == "Congestion Cost":
            color = colormap(cost)  # Get color based on value
        else:
            color = colormap(min_value)
        folium.PolyLine(locations=locations, color=color, weight=5).add_to(m)

st_data = st_folium(m, width=725)