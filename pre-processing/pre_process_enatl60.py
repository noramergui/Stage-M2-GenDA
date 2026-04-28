import numpy as np
import xarray as xr
import glob
import json

data_dir = '/data1/data/models/eNATL60/BLB002/degraded_20/'
output_dir = '/home/nora/GenDA/input_data_enatl60/'

lat_min, lat_max = 28, 38
lon_min, lon_max = -33, -23

# Fichiers SSH uniquement
files = sorted(glob.glob(data_dir + '*_SSH.nc'))
print(f'{len(files)} fichiers SSH trouvés')
print(f'Premier : {files[0]}')
print(f'Dernier : {files[-1]}')

# Ouvrir en dataset multi-temps
print('Chargement...')
ds = xr.open_mfdataset(files, combine='by_coords')
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
print('Dimensions après sélection:', ds.dims)

# Renommer ssh → zos pour compatibilité GenDA
ds = ds.rename({'ssh': 'zos'})

# Charger en mémoire
print('Chargement en mémoire...')
ds = ds.load()

# Moyenne temporelle
print('Calcul des moyennes...')
ds_mean = ds.mean(dim='time')

# Climatologie mensuelle
print('Calcul de la climatologie...')
ds_clim = ds.groupby('time.month').mean('time')

# Sauvegarder
print('Sauvegarde dataset principal...')
ds.to_netcdf(output_dir + 'enatl60_pre_processed.nc')

print('Sauvegarde moyennes...')
ds_mean.to_netcdf(output_dir + 'enatl60_means.nc')

print('Sauvegarde climatologie...')
ds_clim.to_netcdf(output_dir + 'enatl60_climatology.nc')

# Rescale factor = std de l'anomalie
ssh_anom = ds['zos'] - ds_mean['zos']
ssh_std = float(ssh_anom.std())
rescale_factors = {'zos': ssh_std}
print(f'\nRescale factor SSH : {ssh_std:.4f} m')

with open(output_dir + 'diffusion_training_rescale_factors.json', 'w') as f:
    json.dump(rescale_factors, f)

print('\nPré-traitement terminé.')
