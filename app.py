import json
import os
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from streamlit_folium import st_folium

load_dotenv()

@st.cache_data
def load_base_data() -> pd.DataFrame:
    transmission_fp = "data/ne_iso_transmission.geojson"
    with open(transmission_fp) as f:
        gj = json.load(f)

    geo_df = gpd.GeoDataFrame.from_features(gj["features"])
    geo_df = geo_df.loc[geo_df["VOLTAGE"] > 0]

    pnodes_fp = "data/ne_iso_pnodes.csv"
    pnodes_df = pd.read_csv(pnodes_fp)

    base_df = geo_df.merge(pnodes_df, left_on="SUB_1", right_on="Substation Long Name", how="inner")
    base_df["Node/Unit ID"] = base_df["Node/Unit ID"].astype(int)

    return base_df

@st.cache_data
def query_ne_iso_api(endpoint: str) -> dict:
    base_url = 'https://webservices.iso-ne.com/api/v1.1'
    url = base_url + endpoint
    username = os.getenv("NE_ISO_EMAIL")
    password = os.getenv("NE_ISO_PASSWORD")
    headers = {
        'Accept': 'application/json'
    }

    # Make the GET request with basic authentication
    response = requests.get(url, auth=HTTPBasicAuth(username, password), headers=headers)

    if not response.status_code == 200:
        print(f"Failed with status code: {response.status_code}")
        print(response.content)
        return {}

    return response.json()


@st.cache_data
def query_congestion(day: Optional[date]) -> pd.DataFrame:
    if day:
        day_str = day.strftime("%Y%m%d")
    else:
        day_str = "current"

    endpoint = f"/hourlylmp/rt/final/day/{day_str}"
    data = query_ne_iso_api(endpoint)

    if data == {}:
        st.warning("No data returned for congestion cost data.")
        return pd.DataFrame()

    rows = []
    for record in data["HourlyLmps"]["HourlyLmp"]:
        row = {
            "date": record["BeginDate"],
            "loc_id": record["Location"]["@LocId"],
            "lmp": record["LmpTotal"],
            "congestion": record["CongestionComponent"],
            "energy": record["EnergyComponent"],
            "loss": record["LossComponent"]
        }
        rows.append(row)

    lmp_df = pd.DataFrame(rows)
    lmp_df["date"] = pd.to_datetime(lmp_df["date"])
    lmp_df["loc_id"] = lmp_df["loc_id"].astype(int)
    day_lmp_df = lmp_df.groupby([pd.Grouper(key='date', freq='D'), lmp_df.loc_id]).mean().reset_index()

    return day_lmp_df

@st.cache_data
def query_capability(day: Optional[date]) -> pd.DataFrame:
    if day:
        day_str = day.strftime("%Y%m%d")
    else:
        day_str = "current"

    endpoint = f"/totaltransfercapability/day/{day_str}"
    data = query_ne_iso_api(endpoint)

    if data == {}:
        st.warning("No data returned for capability data.")
        return pd.DataFrame()

    rows = []
    for record in data["Ttcs"]["Ttc"]:
        for location in record["TtcLocations"]["TtcLocation"]:
            row = {
                "date": record["BeginDate"],
                "loc_id": location["Location"]["@LocId"],
                "import_mw": location["ImportLimitMw"],
                "export_mw": location["ExportLimitMw"]
            }
            rows.append(row)

    ttc_df = pd.DataFrame(rows)
    ttc_df["date"] = pd.to_datetime(ttc_df["date"])
    ttc_df["loc_id"] = ttc_df["loc_id"].astype(int)

    return ttc_df

@st.cache_data
def merge_api_data(layer: str, input_date: date) -> pd.DataFrame:
    global base_df

    if layer == "Congestion Cost":
        api_df = query_congestion(input_date)
    else:
        return base_df
    
    df = api_df.merge(base_df, left_on="loc_id", right_on="Node/Unit ID", how="inner")

    return df


def get_colormap(values: List[float]) -> cm.ColorMap:
    min_value, max_value = min(values), max(values)
    colormap = cm.linear.YlOrRd_09.scale(min_value, max_value)
    return colormap

@st.cache_data
def load_solar_tif() -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    tif = "data/pvout_atlantic.tif"
    src = rasterio.open(tif)
    array = src.read()
    bounds = src.bounds
    bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
    return array, bbox

def in_region(x: int, y: int) -> bool:
    top_left = [45.085031, -73.539937]
    bottom_right = [41.205254, -70.313054]
    return (
        y > top_left[1] and 
        y < bottom_right[1] and
        x < top_left[0] and
        x > bottom_right[0]
    )

# Base page info
st.title("Energy Generation Planning App")

# Load static data
base_df = load_base_data()

layer = st.selectbox("Feature", [None, "Congestion Cost", "Capacity"])

if layer == "Congestion Cost":
    yesterday = datetime.now() - timedelta(days=1)
    input_date = st.date_input("Date", value=yesterday)
else:
    input_date = None

# Load api data
df = merge_api_data(layer, input_date)

# Initialize a Folium map
m = folium.Map(location=[42.140311, -72.604366], zoom_start=7, prefer_canvas=True)

# Add the colormap legend to the map
if layer == "Congestion Cost":
    color_values = df["lmp"].tolist()
elif layer == "Capacity":
    color_values = df["VOLTAGE"].tolist()
else:
    color_values = None

if color_values:
    colormap = get_colormap(color_values)
    colormap.add_to(m)

for _, row in df.iterrows():
    linestrings = row["geometry"]
    name = row["ID"]
    sub1 = row["SUB_1"]
    sub2 = row["SUB_2"]
    volts = row["VOLTAGE"]

    for i, linestring in enumerate(linestrings.geoms):
        locations = np.flip(np.stack(linestring.xy, axis=1), axis=1)

        if in_region(*locations[0]):

            if layer == "Congestion Cost":
                cost = row["lmp"]
                color = colormap(cost)  # Get color based on value
                layer_val = f"${round(cost, 2)}"
            elif layer == "Capacity":
                color = colormap(volts)
                layer_val = f"{volts} kV"
            else:
                color = "blue"
                layer_val = None

            line = folium.PolyLine(locations=locations, color=color, weight=3)

            if layer_val:
                tooltip = folium.Tooltip(f"{layer} = {layer_val}")
                line.add_child(tooltip)

            line.add_to(m)

            # Add a substation point to the start of the PolyLine if it is the first line
            if i == 0:
                start_point = folium.CircleMarker(
                    location=locations[0],  # Start point coordinates
                    radius=3,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.7,
                    tooltip=sub1
                )
                start_point.add_to(m)

            # Add a substation point at the end of the PolyLine if it is the last line
            if i == (len(linestrings.geoms) - 1) and len(locations) > 0 and len(locations[-1]) == 2:
                end_point = folium.CircleMarker(
                    location=locations[-1],  # End point coordinates
                    radius=2,
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.7,
                )
                end_point.add_to(m)

st_data = st_folium(m, width=725)