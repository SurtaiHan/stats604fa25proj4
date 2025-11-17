import os
import pandas as pd
import numpy as np
import requests
import gzip
import pgeocode
from meteostat import Point, Hourly
import datetime
import pathlib
import holidays
import zoneinfo

def getWeatherFilePath(load_area):
    filepath = "data/weather/" + load_area + ".csv"
    return filepath

def getPjmFilePath(year):
    filepath = "data/pjm/" + "hrl_load_metered_" + str(year) + ".csv"
    return filepath

def getPjmFreshFilePath():
    return "data/pjm/hrl_load_metered_fresh.csv"

def getCompleteDfFilePath(load_area):
    filepath = "data/complete_dfs/" + load_area + ".csv"
    return filepath

def getModelFilePath(load_area):
    filepath = "models/" + load_area + ".pkl"

def makeDirectories(verbose=False):
    base = pathlib.Path("data")
    subdirs = ["complete_dfs", "pjm", "weather"]
    for sub in subdirs:
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created or already existed: {path}")
    path = pathlib.Path("models")
    path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Created or already existed: {path}")

def getLoadAreaToZips():
    # RTO is the *entire PJM footprint*, not a load zone. no meaningful ZIP.
    mymap = {
        # Atlantic City Electric (Southern New Jersey)
        'AECO': ['08401'],   # Atlantic City, NJ
    
        # American Electric Power - Appalachian Power (central and Southern West Virginia)
        'AEPAPT': ['25301'],   # Charleston, WV

        # American Electric Power - Indiana Michigan Power (northeast quadrant of indiana and southwest corner of michigan)
        'AEPIMP': ['46802'],   # Fort Wayne, IN

        # American Electric Power - Kentucky Power (eastern kentucky)
        'AEPKPT': ['41101'],   # Ashland, KY (Eastern Kentucky Power region)

        # American Electric Power - Ohio (central and southeast ohio)
        'AEPOPT': ['43215'],   # Columbus, OH

        # Allegheny Power (FirstEnergy West) serving MD/WV/PA panhandle
        'AP': ['21502'],     # Cumberland, MD

        # Baltimore Gas & Electric
        'BC': ['21201'],     # Baltimore, MD (city center)

        # Cleveland Electric Illuminating Company (FirstEnergy)
        'CE': ['44114'],     # Cleveland, OH (downtown)

        # Dayton Power & Light (AES Ohio)
        'DAY': ['45402'],    # Dayton, OH

        # Duke Energy Ohio/Kentucky load zone
        'DEOK': ['45202'],   # Cincinnati, OH

        # Dominion Virginia Power
        'DOM': ['23219'],    # Richmond, VA

        # Delmarva Power (Delaware & Eastern Shore MD)
        'DPLCO': ['19901'],  # Dover, DE

        # Duquesne Light
        'DUQ': ['15222'],    # Pittsburgh, PA

        # Easton Utilities (Maryland municipal)
        'EASTON': ['21601'], # Easton, MD

        # East Kentucky Power Cooperative
        'EKPC': ['40391'],   # Winchester, KY

        # Jersey Central Power & Light (FirstEnergy NJ)
        'JC': ['07728'],     # Freehold, NJ

        # Metropolitan Edison (FirstEnergy PA)
        'ME': ['19601'],     # Reading, PA

        # Ohio Edison (FirstEnergy OH)
        'OE': ['44308'],     # Akron, OH

        # Ohio Valley Electric Corporation
        'OVEC': ['45661'],   # Piketon, OH

        # Pennsylvania Power Company (FirstEnergy PA)
        'PAPWR': ['16101'],  # New Castle, PA

        # PECO (Philadelphia)
        'PE': ['19103'],     # Philadelphia, PA (Center City)

        # Potomac Electric Power Company (Washington DC + Montgomery Co MD)
        'PEPCO': ['20001'],  # Washington, DC

        # Potomac Edison (FirstEnergy MD/WV)
        'PLCO': ['21740'],   # Hagerstown, MD

        # Pennsylvania Electric Company (Penelec - FirstEnergy Northwest/Central PA)
        'PN': ['16601'],     # Altoona, PA

        # Public Service Electric & Gas (PSE&G NJ)
        'PS': ['07102'],     # Newark, NJ

        # Rockland Electric (Northern NJ / small NY portion)
        'RECO': ['07450'],   # Ridgewood, NJ

        # Southern Maryland Electric Cooperative
        'SMECO': ['20650'],  # Leonardtown, MD

        # UGI Electric (NE Pennsylvania, Luzerne County)
        'UGI': ['18702'],    # Wilkes-Barre, PA

        # VMEU = Virginia Municipal Electric Utility (multiple small cities)
        'VMEU': ['22801']    # Harrisonburg, VA (representative muni)
    }
    return mymap

# Returns the id of the load area (1-indexed)
def getLoadAreaId(load_area):
    load_areas = getLoadAreaToZips().keys()
    load_areas = sorted(load_areas)
    return load_areas.index(load_area)+1


def checkInconsistencies():
    load_area_to_zips = getLoadAreaToZips()
    for year in range(2016,2026):
        df = pd.read_csv(getPjmFilePath(year))
        #print(df['zone'].unique())
        #print(df['load_area'].unique())
        #print(len(df['load_area'].unique()))
        set1 = set(df['load_area'].unique())
        set2 = set(load_area_to_zips.keys())
        set2.add("RTO")
        # Note that years 2016 and 2017 do not match. Thus, we will start with 2018 inclusive
        if len(set1.symmetric_difference(set2)) > 0:
            print(str(year) + " did not match! The difference is: " + str(set1.symmetric_difference(set2)))
    
    print("Finished checking inconsistencies!")

def getYears():
    return range(2018,2026)

def getWeatherDf(load_area, force_refresh=False, verbose=False):
    # Create a Nominatim geocoder instance for the desired country (For the USA, use 'us')
    nomi = pgeocode.Nominatim('us')
    years = getYears()
    # First check if we have the information cached
    file_path = getWeatherFilePath(load_area)
    if os.path.exists(file_path) and not force_refresh:
        temp = pd.read_csv(file_path)
        temp['time'] = pd.to_datetime(temp['time'])
        temp = temp.set_index('time')
        if temp.index.min().year == years[0] and temp.index.max().year == years[-1]:
            if verbose:
                print("Cached weather file for " + load_area + " was found")
            return temp

    # Query the postal code
    load_area_to_zips = getLoadAreaToZips()
    zip_code = load_area_to_zips[load_area][0]
    location_data = nomi.query_postal_code(zip_code)
    latitude = float(location_data.latitude)
    longitude = float(location_data.longitude)

    location = Point(latitude, longitude)
    start = datetime.datetime(years[0], 1, 1)
    end   = datetime.datetime(years[-1], 12, 31)

    data = Hourly(location, start, end, timezone='UTC').fetch()
    # df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    data.to_csv(getWeatherFilePath(load_area))
    if verbose:
        print("Added new cached weather file for " + load_area + " using zip code:" + str(zip_code))
    return data

def getAllWeatherDfs(verbose=False):
    load_area_to_zips = getLoadAreaToZips()
    for key, value in load_area_to_zips.items():
        load_area = key
        getWeatherDf(load_area, verbose=verbose)


def getUSHolidays():
    years = getYears()
    us_holidays = holidays.US(years=years)
    # Add black friday
    for year in years:
        thanksgiving = [day for day, name in us_holidays.items() if name == "Thanksgiving Day" and day.year == year][0]
        us_holidays[thanksgiving + datetime.timedelta(days=1)] = "Black Friday"
        us_holidays[thanksgiving - datetime.timedelta(days=1)] = "Thanksgiving Eve"

    return us_holidays

def isHoliday(datetime_obj, holidays_map):
    return datetime_obj.date() in holidays_map

def isThanksgiving(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Thanksgiving Day"
    
def isThanksgivingEve(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Thanksgiving Eve"
    
def isBlackFriday(datetime_obj, holidays_map):
    return holidays_map.get(datetime_obj.date()) == "Black Friday"
    
def isWeekend(datetime_obj):
    weekno = datetime.datetime.today().weekday()
    if weekno < 5:
        return False
    return True

# Load PJM data
def getEnergyDf(load_area, force_refresh=False, verbose=False):
    years = getYears()
    load_area_to_zips = getLoadAreaToZips()
    if load_area not in load_area_to_zips.keys():
        if verbose:
            print("Error: load_area(" + load_area + ") not found!")
        return -1

    ret_df = None
    for year in years:
        pjm_df = pd.read_csv(getPjmFilePath(year))
        pjm_df = pjm_df[pjm_df['load_area'] == load_area]
        
        if ret_df is None:
            ret_df = pjm_df
        else:
            ret_df = pd.concat([ret_df, pjm_df])

    # Note: 2025 is hardcoded, but whatever
    if 2025 in years:
        # first check if it is cached
        file_path = getPjmFreshFilePath()
        if not os.path.exists(file_path) or force_refresh:
            url = "https://raw.githubusercontent.com/SurtaiHan/stats604fa25proj4/refs/heads/main/data/pjm/hrl_load_metered_fresh.csv"
            out_path = pathlib.Path(getPjmFreshFilePath())
            response = requests.get(url)
            response.raise_for_status()  # raises if download failed
            out_path.write_bytes(response.content)
            if verbose:
                print(f"Downloaded fresh pjm csv to: {out_path}")
        else:
            if verbose:
                print("Cached pjm fresh file was found")

        if verbose:
            print("Loading fresh data in addition to historical data")
        pjm_df = pd.read_csv(getPjmFreshFilePath())
        pjm_df = pjm_df[pjm_df['load_area'] == load_area]
        if ret_df is None:
            ret_df = pjm_df
        else:
            ret_df = pd.concat([ret_df, pjm_df])

    ret_df['datetime_beginning_utc'] = pd.to_datetime(ret_df['datetime_beginning_utc'], utc=True)
    ret_df = ret_df.set_index('datetime_beginning_utc')
    ret_df = ret_df.sort_index()   # Always a good idea after setting index

    return ret_df

# Last step is to augment the pjm data with:
# 1) weather at that current time  (we just use the ground truth weather)
# 2) previous day's weather
# 3) isHoliday, isThanksgiving, isThanksgivingEve, isBlackFriday, isWeekend
# 4) lagged loads (72 hrs ago, and older)

# The resulting dataframe could be viewed as X,y tuples
# in the sense that everything but 

# weather variables available:
# temp	dwpt	rhum	prcp	snow	wdir	wspd	wpgt	pres	tsun	coco
def add_features(energy_df, weather_df, dropna=True, outer=False):
    """
    df must contain at least:
    - mw (float)
    - temp, dwpt, rhum, wspd, tsun, etc. (weather columns)
    - index is hourly timestamps (pd.DatetimeIndex)
    """
    df = energy_df.copy()

    if outer:
        df = df.join(weather_df[['temp']], how='outer')
    else:
        df = df.join(weather_df[['temp']], how='left')
    
    # --- Calendar Features ---
    df['hour'] = df.index.tz_convert('America/New_York').hour
    df['dayofweek'] = df.index.tz_convert('America/New_York').dayofweek
    df['month'] = df.index.tz_convert('America/New_York').month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    # Holiday features (assuming you already defined isHoliday / isThanksgiving):
    us_holidays = getUSHolidays()
    df['is_holiday'] = df.index.tz_convert('America/New_York').to_series().apply(isHoliday, args=(us_holidays,))
    df['is_thanksgiving'] = df.index.tz_convert('America/New_York').to_series().apply(isThanksgiving, args=(us_holidays,))
    df['is_thanksgiving_eve'] = df.index.tz_convert('America/New_York').to_series().apply(isThanksgivingEve, args=(us_holidays,))
    df['is_black_friday'] = df.index.tz_convert('America/New_York').to_series().apply(isBlackFriday, args=(us_holidays,))
    df["is_daylight_savings"] = df.index.tz_convert("America/New_York").to_series().apply(lambda t: t.dst() != pd.Timedelta(0))

    # --- Degree Days ---
    df['HDD'] = (18 - df['temp']).clip(lower=0)
    df['CDD'] = (df['temp'] - 18).clip(lower=0)

    # --- Rolling Weather Averages ---
    df['temp_rolling_24'] = df['temp'].rolling('24h', min_periods=1).mean()
    df['temp_rolling_48'] = df['temp'].rolling('48h', min_periods=1).mean()
    df['temp_rolling_72'] = df['temp'].rolling('72h', min_periods=1).mean()
    # df['dwpt_rolling_24'] = df['dwpt'].rolling(24, min_periods=1).mean()

    # --- Lagged Weather Features ---
    # 24 48 72 96 120 144 168 192 216 240
    df['temp_lag_24'] = df['temp'].shift(freq='24h')
    df['temp_lag_48'] = df['temp'].shift(freq='48h')
    df['temp_lag_72'] = df['temp'].shift(freq='72h')
    df['temp_lag_96'] = df['temp'].shift(freq='96h')
    df['temp_lag_120'] = df['temp'].shift(freq='120h')
    df['temp_lag_144'] = df['temp'].shift(freq='144h')
    df['temp_lag_168'] = df['temp'].shift(freq='168h')
    df['temp_lag_192'] = df['temp'].shift(freq='192h')
    df['temp_lag_216'] = df['temp'].shift(freq='216h')
    df['temp_lag_240'] = df['temp'].shift(freq='240h')

    # --- Lagged Load Features (Key Predictors) ---
    # 72 96 120 144 168 192 216 240
    df['mw_lag_72'] = df['mw'].shift(freq='72h')
    df['mw_lag_96'] = df['mw'].shift(freq='96h')
    df['mw_lag_120'] = df['mw'].shift(freq='120h')
    df['mw_lag_144'] = df['mw'].shift(freq='144h')
    df['mw_lag_168'] = df['mw'].shift(freq='168h')
    df['mw_lag_192'] = df['mw'].shift(freq='192h')
    df['mw_lag_216'] = df['mw'].shift(freq='216h')
    df['mw_lag_240'] = df['mw'].shift(freq='240h')

    #print("Shape before dropna:", df.shape)
    # Drop rows where lag features are missing
    if dropna:
        df = df.dropna()
    #print("Shape after dropna:", df.shape)

    return df

def getCompleteDf(load_area, verbose=False):
    # First check if we have the information cached
    file_path = getCompleteDfFilePath(load_area)
    if os.path.exists(file_path):
        temp = pd.read_csv(file_path)
        temp['datetime_beginning_utc'] = pd.to_datetime(temp['datetime_beginning_utc'], utc=True)
        temp = temp.set_index('datetime_beginning_utc')
        temp = temp.sort_index()
        if verbose:
            print("Cached complete file for " + load_area + " was found")
        return temp

    energy_df = getEnergyDf(load_area, verbose=verbose)
    weather_df = getWeatherDf(load_area, verbose=verbose)
    df_augmented = add_features(energy_df, weather_df)
    # load_area_to_complete_df[load_area] = df_augmented
    df_augmented.to_csv(getCompleteDfFilePath(load_area))
    if verbose:
        print("Added new cached complete file for " + load_area)
    return df_augmented

def getAllCompleteDfs(verbose=False):
    load_area_2_complete_df = {}
    load_area_to_zips = getLoadAreaToZips()
    for load_area in load_area_to_zips.keys():
        load_area_2_complete_df[load_area] = getCompleteDf(load_area, verbose)
    return load_area_2_complete_df

# This function gets the EPT midnight and expresses it as UTC
# calling it with num_days = 0 gives midnight today
# num_days = 1 gives midnight tomorrow, etc
def getMidnight(num_days):
    time_ept = datetime.datetime.now(zoneinfo.ZoneInfo("America/New_York")) + datetime.timedelta(days=num_days)
    midnight_ept = time_ept.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_utc = midnight_ept.astimezone(zoneinfo.ZoneInfo("UTC"))
    return midnight_utc

def getPredictionFeatures(load_area):
    energy_df = getEnergyDf(load_area, force_refresh=True)
    weather_df = getWeatherDf(load_area, force_refresh=True)
    prediction_df = add_features(energy_df, weather_df, dropna=False, outer=True)

    start = getMidnight(1)
    end = getMidnight(2) - datetime.timedelta(hours=1)
    prediction_df = prediction_df.loc[start:end]
    prediction_df["load_area"] = load_area
    prediction_df["datetime_beginning_ept"] = prediction_df.index.tz_convert('America/New_York')
    prediction_df['mw_lag_216'] = prediction_df['mw_lag_216'].fillna(prediction_df['mw_lag_240'])
    prediction_df['mw_lag_192'] = prediction_df['mw_lag_192'].fillna(prediction_df['mw_lag_216'])
    prediction_df['mw_lag_168'] = prediction_df['mw_lag_168'].fillna(prediction_df['mw_lag_192'])
    prediction_df['mw_lag_144'] = prediction_df['mw_lag_144'].fillna(prediction_df['mw_lag_168'])
    prediction_df['mw_lag_120'] = prediction_df['mw_lag_120'].fillna(prediction_df['mw_lag_144'])
    prediction_df['mw_lag_96'] = prediction_df['mw_lag_96'].fillna(prediction_df['mw_lag_120'])
    prediction_df['mw_lag_72'] = prediction_df['mw_lag_72'].fillna(prediction_df['mw_lag_96'])

    prediction_df['temp_rolling_48'] = prediction_df['temp_rolling_48'].fillna(prediction_df['temp_rolling_72'])
    prediction_df['temp_rolling_24'] = prediction_df['temp_rolling_24'].fillna(prediction_df['temp_rolling_48'])

    prediction_df['temp_lag_216'] = prediction_df['temp_lag_216'].fillna(prediction_df['temp_lag_240'])
    prediction_df['temp_lag_192'] = prediction_df['temp_lag_192'].fillna(prediction_df['temp_lag_216'])
    prediction_df['temp_lag_168'] = prediction_df['temp_lag_168'].fillna(prediction_df['temp_lag_192'])
    prediction_df['temp_lag_144'] = prediction_df['temp_lag_144'].fillna(prediction_df['temp_lag_168'])
    prediction_df['temp_lag_120'] = prediction_df['temp_lag_120'].fillna(prediction_df['temp_lag_144'])
    prediction_df['temp_lag_96'] = prediction_df['temp_lag_96'].fillna(prediction_df['temp_lag_120'])
    prediction_df['temp_lag_72'] = prediction_df['temp_lag_72'].fillna(prediction_df['temp_lag_96'])

    return prediction_df
