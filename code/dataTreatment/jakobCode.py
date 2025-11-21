import json
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

path = 'data/wyndham_smartbin_filllevel.json'

# Load
with open(path, 'r') as f:
    json_object = json.load(f)

rows = []
for feature in json_object["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    rows.append({
        "timestamp": props["timestamp"],
        "fullnessthreshold": props["fullnessThreshold"],
        "serialnumber": props["serialNumber"],
        "latestfullness": props["latestFullness"],
        "coordinates": coords
    })

df = pd.DataFrame(rows)
print(df.head())



timestamp_counts = df['timestamp'].value_counts()
print("Unique timestamps and their counts:")
print(timestamp_counts.head(10))

#Uncomment if you want to check how many bins there are and how many samples are present
print("Unique Serialnumbers:")
print(df["serialnumber"].unique())
print(df["serialnumber"].value_counts())
len(df["serialnumber"].unique())
