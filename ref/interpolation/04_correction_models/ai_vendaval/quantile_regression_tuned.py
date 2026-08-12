import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
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
    
    rename_map = {
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        '2m_temperature': 't2m',
        '2m_dewpoint_temperature': 'd2m',
        'mean_sea_level_pressure': 'msl',
        'total_precipitation': 'tp'
    }
    
    actual_rename = {k: v for k, v in rename_map.items() if k in ds_era}
    ds_era = ds_era.rename(actual_rename)
    
    features = []
    
    if 'u10' in ds_era and 'v10' in ds_era:
        ws_h = np.sqrt(ds_era.u10**2 + ds_era.v10**2)
        wd_h = (np.degrees(np.arctan2(-ds_era.u10, -ds_era.v10)) + 360) % 360
        features.append(ws_h.resample(time="1D").mean().rename("ws_mean"))
        features.append(ws_h.resample(time="1D").max().rename("ws_max"))
        features.append(ws_h.resample(time="1D").std().rename("ws_std"))
        features.append(ws_h.resample(time="1D").quantile(0.9).drop_vars("quantile").rename("ws_p90"))
        features.append(np.sin(np.radians(wd_h)).resample(time="1D").mean().rename("sin_dir_mean"))
        features.append(np.cos(np.radians(wd_h)).resample(time="1D").mean().rename("cos_dir_mean"))
        
    if 'msl' in ds_era:
        features.append(ds_era.msl.resample(time="1D").mean().rename("msl_mean"))
        features.append(ds_era.msl.resample(time="1D").min().rename("msl_min"))
        features.append((ds_era.msl.resample(time="1D").max() - ds_era.msl.resample(time="1D").min()).rename("msl_range"))
        
    if 't2m' in ds_era:
        features.append(ds_era.t2m.resample(time="1D").mean().rename("t2m_mean"))
        features.append(ds_era.t2m.resample(time="1D").max().rename("t2m_max"))
        features.append(ds_era.t2m.resample(time="1D").min().rename("t2m_min"))
        features.append((ds_era.t2m.resample(time="1D").max() - ds_era.t2m.resample(time="1D").min()).rename("t2m_range"))
        
    if 't2m' in ds_era and 'd2m' in ds_era:
        t_c = ds_era.t2m - 273.15
        td_c = ds_era.d2m - 273.15
        a = 17.625
        b = 243.04
        rh_h = 100 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t_c) / (b + t_c))
        
        features.append(rh_h.resample(time="1D").mean().rename("rh_mean"))
        features.append(rh_h.resample(time="1D").quantile(0.1).drop_vars("quantile").rename("rh_p10"))
        features.append((ds_era.t2m - ds_era.d2m).resample(time="1D").mean().rename("td_dep_mean"))
        
    if 'tp' in ds_era:
        features.append(ds_era.tp.resample(time="1D").sum().rename("tp_sum"))
        features.append(ds_era.tp.resample(time="1D").max().rename("tp_max"))
        
    ds_era_daily = xr.merge(features)
    
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
    # Sort by time for TimeSeriesSplit
    df = df.sort_values(by='time').reset_index(drop=True)
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
    
    X = df_clean[selected_features]
    y = df_clean[target]
    
    print("Setting up Pipeline for Tuning...")
    # Pipeline: StandardScaler -> PolynomialFeatures -> QuantileRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        # Using solver 'highs' if available, otherwise interior-point
        ('regressor', QuantileRegressor(quantile=0.95))
    ])
    
    # Grid Search params
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 1.0] # Tuning L1 penalty
    }
    
    print("Starting GridSearchCV with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid = GridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    print(f"Best parameters: {grid.best_params_}")
    
    best_model = grid.best_estimator_
    
    print("Predicting and evaluating...")
    y_pred = best_model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    print(f"Tuned Metrics: R2={r2:.3f}, RMSE={rmse:.3f}, MSE={mse:.3f}")
    
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
    axes[1].scatter(y_pred, y, alpha=0.3, color='dodgerblue')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (y=x)')
    
    axes[1].set_title(f'Tuned Quantile Reg. (q=0.95)\nActual vs Predicted', fontsize=14)
    axes[1].set_xlabel('Predicted Wind Gust (m/s)', fontsize=12)
    axes[1].set_ylabel('Actual Wind Gust (m/s)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    metrics_text = f"Best alpha: {grid.best_params_['regressor__alpha']}\nRMSE: {rmse:.3f}\nMSE: {mse:.3f}\n$R^2$: {r2:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    axes[1].legend(loc='lower right')
    
    # Plot 3: Residuals
    residuals = y - y_pred
    axes[2].scatter(y_pred, residuals, alpha=0.3, color='darkorange')
    axes[2].axhline(y=0, color='red', linestyle='--', label='Zero Residual')
    
    axes[2].set_title('Residuals Scatterplot\n(Actual - Predicted) vs Predicted', fontsize=14)
    axes[2].set_xlabel('Predicted Wind Gust (m/s)', fontsize=12)
    axes[2].set_ylabel('Residuals (m/s)', fontsize=12)
    axes[2].grid(True, linestyle=':', alpha=0.7)
    axes[2].legend()
    
    plt.tight_layout()
    combined_plot_path = os.path.join(out_dir, 'quantile_regression_tuned_combined.png')
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    
    print("Done! Tuned combined plot saved successfully.")

if __name__ == "__main__":
    main()
