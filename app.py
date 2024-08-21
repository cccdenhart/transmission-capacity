import json
import os
from typing import List

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

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
        .mean(["congestion_price_rt", "VOLTAGE"])
        .reset_index()
    )
    avg_geo_cost_df = avg_geo_cost_df.loc[avg_geo_cost_df["VOLTAGE"] > 0]

    return avg_geo_cost_df

@st.cache_data
def load_substation_data() -> gpd.GeoDataFrame:
    substation_fp = "data/Substations.csv"
    substation_df = pd.read_csv(substation_fp)
    substation_gdf = gpd.GeoDataFrame(
        substation_df, geometry=gpd.points_from_xy(substation_df.LONGITUDE, substation_df.LATITUDE), crs="EPSG:4326"
    )

    return substation_gdf



def get_colormap(values: List[float]) -> cm.ColorMap:
    min_value, max_value = min(values), max(values)
    colormap = cm.linear.YlOrRd_09.scale(min_value, max_value)
    return colormap

# Base page info
st.title("Energy Generation Planning App")

layer = st.selectbox("Feature", [None, "Congestion Cost", "Capacity"])

# Load data
df = load_data()

# Load substation data
substation_gdf = load_substation_data()

# Initialize a Folium map
m = folium.Map(location=[39.653806, -77.152707], zoom_start=7, prefer_canvas=True,)

# Add the colormap legend to the map
if layer == "Congestion Cost":
    color_values = df["congestion_price_da"].tolist()
elif layer == "Capacity":
    color_values = df["VOLTAGE"].tolist()
else:
    color_values = None

if color_values:
    colormap = get_colormap(color_values)
    colormap.add_to(m)

gdf = gpd.GeoDataFrame(df , geometry = df.geometry)
gdf.crs="EPSG:4326"

folium.Choropleth(
    gdf,
    line_weight=3,
    line_color='blue'
).add_to(m)

marker_cluster = MarkerCluster(
    locations=substation_gdf[['LATITUDE', 'LONGITUDE']].values.tolist(),
    name="transmission network substations",
    overlay=True,
    control=True,
)

marker_cluster.add_to(m)

st_data = st_folium(m, width=725)