import xarray as xr
import pandas as pd
import numpy as np
import os

# Paths
CSV_PATH = "/home/publico/vendaval/data/in/Training_Dataset_INMET_ERA5_Paired.csv"
NC_PATH = "/home/publico/vendaval/data/in/ERA5_Features_ROI_2000_2025.nc"
OUTPUT_PATH = "/home/publico/vendaval/data/out/ERA5_INMET_Paired_ROI_4_Stations.nc"

# Selected Stations
STATIONS = {
    'A805': (-27.8544, -53.7911),
    'A853': (-28.6033, -53.6736),
    'A854': (-27.3956, -53.4294),
    'A856': (-27.9203, -53.3181)
}

def main():
    print("Loading ERA5 Features...")
    ds_era5 = xr.open_dataset(NC_PATH)
    
    print("Loading INMET Targets (CSV)...")
    # Only read columns we need to save memory
    df_targets = pd.read_csv(CSV_PATH, usecols=['data', 'codigo_estacao', 'rajada_max_final_ms'])
    df_targets['data'] = pd.to_datetime(df_targets['data'])
    
    station_datasets = []
    
    for station_code, (lat, lon) in STATIONS.items():
        print(f"Processing station {station_code} at ({lat}, {lon})...")
        
        # 1. Extract nearest ERA5 point
        # Use selection with nearest neighbor
        ds_point = ds_era5.sel(latitude=lat, longitude=lon, method='nearest')
        
        # 2. Get INMET targets for this station
        df_station = df_targets[df_targets['codigo_estacao'] == station_code].copy()
        
        # 3. Align time series
        # Convert to xarray DataArray
        target_da = xr.DataArray(
            df_station['rajada_max_final_ms'].values,
            coords={'time': df_station['data'].values},
            dims=['time'],
            name='rajada_max_final_ms'
        ).sortby('time')
        
        # 4. Merge
        # Inner join on time to ensure we have both features and targets
        ds_combined = xr.merge([ds_point, target_da], join='inner')
        
        # Add metadata
        ds_combined = ds_combined.expand_dims(station=[station_code])
        ds_combined.coords['station_lat'] = (('station'), [lat])
        ds_combined.coords['station_lon'] = (('station'), [lon])
        
        station_datasets.append(ds_combined)
    
    print("Combining all stations...")
    ds_final = xr.concat(station_datasets, dim='station')
    
    print(f"Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    ds_final.to_netcdf(OUTPUT_PATH)
    
    print("Success!")
    print(ds_final)

if __name__ == "__main__":
    main()
