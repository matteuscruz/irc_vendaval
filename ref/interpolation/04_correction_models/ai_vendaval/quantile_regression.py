import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    nc_path = '/home/publico/vendaval/02_preprocessing/era_inmet_cluster_4.nc'
    out_dir = '/home/publico/vendaval/04_correction_models/ai_vendaval'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("Loading data...")
    ds = xr.open_dataset(nc_path)
    
    era_vars_orig = [v for v in ds.data_vars if 'time_era5' in ds[v].dims]
    inmet_vars = [v for v in ds.data_vars if 'time_inmet' in ds[v].dims]
    
    print("Applying Feature Engineering to ERA5 data...")
    ds_era = ds[era_vars_orig].rename({'time_era5': 'time'})
    
    # Mapping variables
    rename_map = {
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        '2m_temperature': 't2m',
        '2m_dewpoint_temperature': 'd2m',
        'mean_sea_level_pressure': 'msl',
        'total_precipitation': 'tp'
    }
    
    # Only rename the ones that exist
    actual_rename = {k: v for k, v in rename_map.items() if k in ds_era}
    ds_era = ds_era.rename(actual_rename)
    
    features = []
    
    # 1. Vento
    if 'u10' in ds_era and 'v10' in ds_era:
        ws_h = np.sqrt(ds_era.u10**2 + ds_era.v10**2)
        wd_h = (np.degrees(np.arctan2(-ds_era.u10, -ds_era.v10)) + 360) % 360
        features.append(ws_h.resample(time="1D").mean().rename("ws_mean"))
        features.append(ws_h.resample(time="1D").max().rename("ws_max"))
        features.append(ws_h.resample(time="1D").std().rename("ws_std"))
        features.append(ws_h.resample(time="1D").quantile(0.9).drop_vars("quantile").rename("ws_p90"))
        features.append(np.sin(np.radians(wd_h)).resample(time="1D").mean().rename("sin_dir_mean"))
        features.append(np.cos(np.radians(wd_h)).resample(time="1D").mean().rename("cos_dir_mean"))
        
    # 2. Pressão
    if 'msl' in ds_era:
        features.append(ds_era.msl.resample(time="1D").mean().rename("msl_mean"))
        features.append(ds_era.msl.resample(time="1D").min().rename("msl_min"))
        features.append((ds_era.msl.resample(time="1D").max() - ds_era.msl.resample(time="1D").min()).rename("msl_range"))
        
    # 3. Temperatura
    if 't2m' in ds_era:
        features.append(ds_era.t2m.resample(time="1D").mean().rename("t2m_mean"))
        features.append(ds_era.t2m.resample(time="1D").max().rename("t2m_max"))
        features.append(ds_era.t2m.resample(time="1D").min().rename("t2m_min"))
        features.append((ds_era.t2m.resample(time="1D").max() - ds_era.t2m.resample(time="1D").min()).rename("t2m_range"))
        
    # 4. Umidade
    if 't2m' in ds_era and 'd2m' in ds_era:
        t_c = ds_era.t2m - 273.15
        td_c = ds_era.d2m - 273.15
        a = 17.625
        b = 243.04
        rh_h = 100 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t_c) / (b + t_c))
        
        features.append(rh_h.resample(time="1D").mean().rename("rh_mean"))
        features.append(rh_h.resample(time="1D").quantile(0.1).drop_vars("quantile").rename("rh_p10"))
        features.append((ds_era.t2m - ds_era.d2m).resample(time="1D").mean().rename("td_dep_mean"))
        
    # 5. Precipitação
    if 'tp' in ds_era:
        features.append(ds_era.tp.resample(time="1D").sum().rename("tp_sum"))
        features.append(ds_era.tp.resample(time="1D").max().rename("tp_max"))
        
    ds_era_daily = xr.merge(features)
    
    # Rolling and Lags
    if 'msl_mean' in ds_era_daily:
        ds_era_daily["d_msl"] = ds_era_daily.msl_mean.diff(dim="time", label="upper")
    if 'tp_sum' in ds_era_daily:
        ds_era_daily["tp_lag1"] = ds_era_daily.tp_sum.shift(time=1)
        ds_era_daily["tp_lag2"] = ds_era_daily.tp_sum.shift(time=2)
        ds_era_daily["tp_roll3"] = ds_era_daily.tp_sum.rolling(time=3).sum()
    
    engineered_vars = list(ds_era_daily.data_vars.keys())
    
    print("Preparing INMET data...")
    ds_inmet = ds[inmet_vars].rename({'time_inmet': 'time'})
    
    print("Converting to pandas DataFrames...")
    df_era = ds_era_daily.to_dataframe().reset_index()
    df_inmet = ds_inmet.to_dataframe().reset_index()
    
    df_era['time'] = pd.to_datetime(df_era['time']).dt.normalize()
    df_inmet['time'] = pd.to_datetime(df_inmet['time']).dt.normalize()
    df_era['estacao'] = df_era['estacao'].astype(str)
    df_inmet['estacao'] = df_inmet['estacao'].astype(str)
    
    print("Merging datasets...")
    df = pd.merge(df_inmet, df_era, on=['time', 'estacao'], how='inner')
    print(f"Data merged. Total rows: {len(df)}")
    
    target = 'daily_wind_gust_max'
    
    print("Calculating predep (|Pearson r|)...")
    corrs = df[engineered_vars + [target]].corr()[target].drop(target).abs()
    selected_features = corrs[corrs > 0.15].index.tolist()
    
    print(f"Features with predep > 0.15: {selected_features}")
    
    if len(selected_features) == 0:
        best_feat = corrs.idxmax()
        selected_features = [best_feat]
        print(f"No features > 0.15. Using best feature: {best_feat}")
        
    df_clean = df.dropna(subset=[target] + selected_features).copy()
    print(f"Rows after dropping NaNs: {len(df_clean)}")
    
    print("Building formula...")
    formula_parts = []
    for feat in selected_features:
        formula_parts.append(f'Q("{feat}") + I(Q("{feat}") ** 2.0)')
    formula = f'Q("{target}") ~ ' + ' + '.join(formula_parts)
    print(f"Formula: {formula}")
    
    print("Fitting Quantile Regression for EXTREMES (q=0.95)...")
    mod = smf.quantreg(formula, df_clean)
    res = mod.fit(q=0.95, max_iter=2000)
    print(res.summary())
    
    print("Predicting and evaluating...")
    y_pred = res.predict(df_clean)
    y_real = df_clean[target]
    
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    
    print(f"Metrics: R2={r2:.3f}, RMSE={rmse:.3f}, MSE={mse:.3f}")
    
    print("Generating combined plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Predep Variable Selection
    corrs_sorted = corrs.sort_values(ascending=True)
    colors = ['limegreen' if val > 0.15 else 'lightcoral' for val in corrs_sorted]
    
    axes[0].barh(corrs_sorted.index, corrs_sorted.values, color=colors, edgecolor='black', alpha=0.8)
    axes[0].axvline(x=0.15, color='red', linestyle='--', label='Threshold (0.15)')
    axes[0].set_title('Predep Variable Selection (|Pearson r|)', fontsize=14)
    axes[0].set_xlabel('Predep (|r|)', fontsize=12)
    axes[0].grid(axis='x', linestyle=':', alpha=0.7)
    axes[0].legend()
    
    # Plot 2: Actual vs Predicted
    axes[1].scatter(y_pred, y_real, alpha=0.3, color='dodgerblue')
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (y=x)')
    
    axes[1].set_title(f'Quantile Regression (q=0.95)\nActual vs Predicted Wind Gust', fontsize=14)
    axes[1].set_xlabel('Predicted Wind Gust (m/s)', fontsize=12)
    axes[1].set_ylabel('Actual Wind Gust (m/s)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    metrics_text = f"RMSE: {rmse:.3f}\nMSE: {mse:.3f}\n$R^2$: {r2:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    axes[1].legend(loc='lower right')
    
    # Plot 3: Residuals
    residuals = y_real - y_pred
    axes[2].scatter(y_pred, residuals, alpha=0.3, color='darkorange')
    axes[2].axhline(y=0, color='red', linestyle='--', label='Zero Residual')
    
    axes[2].set_title('Residuals Scatterplot\n(Actual - Predicted) vs Predicted', fontsize=14)
    axes[2].set_xlabel('Predicted Wind Gust (m/s)', fontsize=12)
    axes[2].set_ylabel('Residuals (m/s)', fontsize=12)
    axes[2].grid(True, linestyle=':', alpha=0.7)
    axes[2].legend()
    
    plt.tight_layout()
    combined_plot_path = os.path.join(out_dir, 'quantile_regression_combined.png')
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    
    print("Done! Combined plot saved successfully.")

if __name__ == "__main__":
    main()
