

import pandas as pd
import requests
import gzip
#from isdparser import isdparser
from meteostat import Point, Hourly
from datetime import datetime


"""usaf = "722950"
wban = "23174"
year = "2024"

url = f"https://www.ncei.noaa.gov/pub/data/noaa/{year}/{usaf}-{wban}-{year}.gz"
response = requests.get(url)

with open("station-raw.gz", "wb") as f:
    f.write(response.content)

records = isdparser.ISDParser("station-raw.gz").records
df = pd.DataFrame(records)
df.head()

stations = pd.read_csv("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv")
# Filter to stations of interest (e.g., by ICAO code like KPHL, KBWI, KCMH etc.)
stations = stations[stations['ICAO'] == 'KCMH']
print(stations)

years = range(2024, 2025)
for y in years:
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/{y}/{usaf}-{wban}-{y}.gz"
    r = requests.get(url)
    open(f"{usaf}-{wban}-{y}.gz", "wb").write(r.content)"""

# Example: Columbus airport coordinates
location = Point(40.0, -82.9)  # use exact lat/lon for station if desired
start = datetime(2024, 1, 1)
end   = datetime(2024, 12, 31)

data = Hourly(location, start, end).fetch()
print(data.head())
