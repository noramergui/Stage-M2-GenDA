import os
import numpy as np
import xarray as xr

# ── Chargement des données GLORYS ──────────────────────────────────────────
ds_g = xr.open_zarr('/data1/data/obs/level4/GLORYS_2010.zarr/')

# ── Traitement par région ──────────────────────────────────────────────────
for region_name, lon_min, lon_max, lat_min, lat_max in [
    ('azores',     -43, -13, 23, 43),
    ('gulfstream', -70, -40, 25, 45),
]:
    
    output_dir = f'/data2/nora/GenDA_workspace/input_data_{region_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # Sélection de la région
    ds_region = ds_g.sel(
        latitude  = slice(lat_min, lat_max),
        longitude = slice(lon_min, lon_max)
    )[['zos']].load()  # SSH uniquement

    # Moyenne temporelle
    ds_m = ds_region.mean(dim='time')
    ds_m.to_netcdf(output_dir + 'glorys_means_pre_processed_fixed_noislands.nc')

    # Climatologie mensuelle
    ds_clim = ds_region.groupby('time.month').mean('time')
    ds_clim.to_netcdf(output_dir + 'glorys_climatology.nc')

    # Dataset principal
    ds_region.to_netcdf(output_dir + 'glorys_pre_processed_fixed_noislands.nc')
    # Rescale factor
    import json
    ssh_anom = ds_region['zos'] - ds_m['zos']
    ssh_std = float(ssh_anom.std())
    with open(output_dir + 'diffusion_training_rescale_factors.json', 'w') as f:
        json.dump({'zos': ssh_std}, f)

    print(f'{region_name} : ssh_std = {ssh_std:.4f} m')