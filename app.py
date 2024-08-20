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
    nrel_wind_lcoe_fp = "data/nrel_lbw_lcoe.csv"
    nrel_solarpv_lcoe_fp = "data/nrel_solarpv_lcoe.csv"


    cost_df = pd.read_csv(cost_fp)
    nrel_wind_lcoe_df = pd.read_csv(nrel_wind_lcoe_fp)
    nrel_solarpv_lcoe_df = pd.read_csv(nrel_solarpv_lcoe_fp)


    geo_cost_df = geo_df.merge(cost_df, left_on="SUB_1", right_on="pnode_name", how="inner")
    avg_geo_cost_df = (
        geo_cost_df.groupby(
            ["ID", "pnode_id", "pnode_name", "SUB_1", "SUB_2", "VOLTAGE", "geometry"]
        )
        .mean("congestion_price_rt")
        .reset_index()
    )

    return avg_geo_cost_df, nrel_solarpv_lcoe_df, nrel_wind_lcoe_df

def load_solar_tif() -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    tif = "data/pvout_atlantic.tif"
    src = rasterio.open(tif)
    array = src.read()
    bounds = src.bounds
    bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
    return array, bbox

# Base page info
st.title("Energy Generation Planning App")

# Load data
geo_df, solar_lcoe, wind_lcoe = load_data()

#Load raster data
pv, bbox = load_solar_tif()

# Initialize a Folium map
m = folium.Map(location=[39.653806, -77.152707], zoom_start=5)

#add groups for each layer

congestion_group = folium.FeatureGroup(name = 'Congestion Data').add_to(m)
solar_lcoe_group = folium.FeatureGroup(name = 'Solar LCOE Data').add_to(m)
# wind_lcoe_group = folium.FeatureGroup(name = 'Wind LCOE Data').add_to(m)

# Add the colormap legend to the map
costs = geo_df["congestion_price_da"]
min_value, max_value = min(costs), max(costs)
colormap_congestion = cm.linear.YlOrRd_09.scale(min_value, max_value)
colormap_congestion.caption = 'Congestion Colormap'
colormap_congestion.add_to(m)

solar_lcoe_values = solar_lcoe["mean_lcoe"]
colormap_solar_lcoe = cm.linear.YlOrRd_09.scale(min(solar_lcoe_values), max(solar_lcoe_values))
colormap_solar_lcoe.caption = 'Mean Solar LCOE Colormap'
colormap_solar_lcoe.add_to(m)

# wind_lcoe_values = wind_lcoe["mean_lcoe"]
# colormap_wind_lcoe = cm.linear.YlOrRd_09.scale(min(wind_lcoe_values), max(wind_lcoe_values))
# colormap_wind_lcoe.caption = 'Mean Wind LCOE Colormap'
# colormap_wind_lcoe.add_to(m)


# ## Congestion Data
for _, row in geo_df.iterrows():
    linestrings = row["geometry"]
    name = row["ID"]
    cost = row["congestion_price_da"]

    for linestring in linestrings.geoms:
        locations = np.flip(np.stack(linestring.xy, axis=1), axis=1)
        color = colormap_congestion(cost)
        folium.PolyLine(locations=locations, color=color, weight=5).add_to(congestion_group)



## Solar LCOE Data
for _,row in solar_lcoe.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    solar_lcoe = row["mean_lcoe"]
    color = colormap_solar_lcoe(solar_lcoe)


    folium.Circle(
        location = [lat, lon],
        color = color,
        fill_color = color,
        radius = 100
    ).add_to(solar_lcoe_group)



# ## Wind LCOE Data
# for _,row in wind_lcoe.iterrows():
#     lat = row["latitude"]
#     lon = row["longitude"]
#     wind_lcoe = row["mean_lcoe"]
#     color = colormap_wind_lcoe(wind_lcoe)


#     folium.Circle(
#         location = [lat, lon],
#         color = color,
#         fill_color = color,
#         radius = 100
#     ).add_to(wind_lcoe_group)

# Add LayerControl to the map
folium.LayerControl().add_to(m)

st_data = st_folium(m, width=725, height= 500)