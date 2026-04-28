import numpy as np
import xarray as xr
import glob
import json
from datetime import date, timedelta

data_dir   = '/data1/data/models/eNATL60/BLB002/'
output_dir = '/home/nora/GenDA/input_data_enatl60/'

lat_min, lat_max = 28, 38
lon_min, lon_max = -33, -23

# Grille interne 128x128 (après buffers de 36)
ds_ref = xr.open_dataset(
    '/data1/data/models/eNATL60/BLB002/degraded_20/eNATL60-BLB002_y2009m07d01.1d_SSH.nc'
).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
lat_grid = ds_ref.lat.values[36:164]  # (128,)
lon_grid = ds_ref.lon.values[36:164]  # (128,)

# Moyenne et rescale factor pour normaliser les obs
ds_mean = xr.open_dataset(output_dir + 'enatl60_means.nc')
ssh_mean_2d = ds_mean['zos'].sel(
    lat=slice(lat_grid[0], lat_grid[-1]),
    lon=slice(lon_grid[0], lon_grid[-1])
).values  # (128, 128)

with open(output_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)
ssh_std = rescale_factors['zos']

# Charger nadirs (filtrer sur la boîte une fois)
print('Chargement des nadirs...')
satellites = ['al', 'c2', 'h2b', 'j3', 's3a', 's3b', 's6a']
ds_nadirs = []
for s in satellites:
    ds = xr.open_dataset(f'{data_dir}obs/nadirs/{s}.nc')
    mask = ((ds.lat >= lat_min) & (ds.lat <= lat_max) &
            (ds.lon >= lon_min) & (ds.lon <= lon_max))
    ds_nadirs.append(ds.isel(time=mask).load())
    print(f'  {s} : {int(mask.sum())} points')

swot_files = sorted(glob.glob(f'{data_dir}obs/swot/*.nc'))
print(f'SWOT : {len(swot_files)} fichiers')

# Traitement jour par jour — période de test juin 2010
dates = [date(2010, 6, 1) + timedelta(days=i) for i in range(30)]

all_obs = {}  # dictionnaire date -> dict avec lats, lons, ssh normalisé

for d in dates:
    t0 = np.datetime64(d)
    t1 = np.datetime64(d + timedelta(days=1))

    lats_day, lons_day, ssh_day = [], [], []

    # Nadirs
    for ds in ds_nadirs:
        mask_t = (ds.time.values >= t0) & (ds.time.values < t1)
        if mask_t.sum() == 0:
            continue
        lats_day.append(ds.lat.values[mask_t])
        lons_day.append(ds.lon.values[mask_t])
        ssh_day.append(ds.ssh.values[mask_t])

    # SWOT
    for f in swot_files:
        ds_s = xr.open_dataset(f)
        if ds_s.time.values.min() > t1 or ds_s.time.values.max() < t0:
            continue
        mask_t = (ds_s.time.values >= t0) & (ds_s.time.values < t1)
        mask_geo = ((ds_s.latitude.values >= lat_grid[0]) &
                    (ds_s.latitude.values <= lat_grid[-1]) &
                    (ds_s.longitude.values >= lon_grid[0]) &
                    (ds_s.longitude.values <= lon_grid[-1]))
        mask_st = mask_t[:, np.newaxis] & mask_geo
        if mask_st.sum() == 0:
            continue
        lats_day.append(ds_s.latitude.values[mask_st])
        lons_day.append(ds_s.longitude.values[mask_st])
        ssh_day.append(ds_s.ssh.values[mask_st])

    if len(lats_day) == 0:
        print(f'{d} : aucune observation')
        all_obs[str(d)] = None
        continue

    lats_all = np.concatenate(lats_day).astype(np.float32)
    lons_all = np.concatenate(lons_day).astype(np.float32)
    ssh_all  = np.concatenate(ssh_day).astype(np.float32)

    # Supprimer NaN
    valid    = ~np.isnan(ssh_all)
    lats_all = lats_all[valid]
    lons_all = lons_all[valid]
    ssh_all  = ssh_all[valid]

    # Normaliser : soustraire la moyenne interpolée au point obs + diviser par std
    # Interpolation bilinéaire de ssh_mean_2d aux positions obs
    from scipy.interpolate import RegularGridInterpolator
    interp_mean = RegularGridInterpolator(
        (lat_grid, lon_grid), ssh_mean_2d, method='linear', bounds_error=False, fill_value=None
    )
    ssh_mean_at_obs = interp_mean(np.column_stack([lats_all, lons_all]))
    ssh_norm = (ssh_all - ssh_mean_at_obs) / ssh_std

    all_obs[str(d)] = {
        'lats': lats_all,
        'lons': lons_all,
        'ssh':  ssh_norm,
        'n':    len(ssh_norm),
    }
    print(f'{d} : {len(ssh_norm)} observations (normalisées, mean={ssh_norm.mean():.3f}, std={ssh_norm.std():.3f})')

# Sauvegarder en numpy
np.save(output_dir + 'obs_enatl60_juin2010.npy', all_obs)
print('\nSauvegardé : obs_enatl60_juin2010.npy')
print('Grille interne sauvegardée.')
np.save(output_dir + 'lat_grid_128.npy', lat_grid)
np.save(output_dir + 'lon_grid_128.npy', lon_grid)
