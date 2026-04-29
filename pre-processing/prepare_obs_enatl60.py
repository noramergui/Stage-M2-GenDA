# imports et paramètres

import numpy as np
import xarray as xr
import glob
import json
from datetime import date, timedelta 

data_dir   = '/data1/data/models/eNATL60/BLB002/'
output_dir = '/home/nora/GenDA/input_data_enatl60/'

lat_min, lat_max = 28, 38
lon_min, lon_max = -33, -23


# Charger la grille de référence 128×128

ds = xr.open_dataset('/data1/data/models/eNATL60/BLB002/degraded_20/eNATL60-BLB002_y2009m07d01.1d_SSH.nc')
ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

buffer = 36 #pixels

lat_grid = ds.lat.values[buffer:-buffer]  # (128,)
lon_grid = ds.lon.values[buffer:-buffer]  # (128,)

print('lat_grid shape:', lat_grid.shape)
print('lon_grid shape:', lon_grid.shape)
print('lat range:', lat_grid[0], '->', lat_grid[-1])
print('lon range:', lon_grid[0], '->', lon_grid[-1])

# Charger la moyenne SSH et le rescale factor
# On en a besoin pour normaliser les observations avant de les comparer 
# avec l'état du modèle x (qui lui est déjà normalisé).

ds_mean = xr.open_dataset('/data2/nora/GenDA_workspace/input_data_enatl60/enatl60_means.nc')
ssh_mean_2d = ds_mean['zos'].sel(
    lat=slice(lat_grid[0], lat_grid[-1]),
    lon=slice(lon_grid[0], lon_grid[-1])
).values  # .values pour convertir en numpy
print('ssh_mean_2d shape:', ssh_mean_2d.shape)

# on récupère l'écart type de la SSH
with open('/data2/nora/GenDA_workspace/input_data_enatl60/diffusion_training_rescale_factors.json', 'r') as f:
    rescale_factors = json.load(f)
ssh_std = rescale_factors['zos']
print('ssh_std:', ssh_std)

# Charger les nadirs
satellites = ['al', 'c2', 'h2b', 'j3', 's3a', 's3b', 's6a']

ds_nadirs = []

for nadir in satellites : 
    ds = xr.open_dataset('/data1/data/models/eNATL60/BLB002/obs/nadirs/' + nadir + '.nc')
    mask_lat = (ds.lat >= lat_grid[0]) & (ds.lat <= lat_grid[-1]) # liste de True/False 
    mask_lon = (ds.lon >= lon_grid[0]) & (ds.lon <= lon_grid[-1]) # liste de True/False 

    mask = mask_lat & mask_lon # liste de True/False 
    
    ds_filtre = ds.isel(time=mask).load() #on garde les points où mask=True
    ds_nadirs.append(ds_filtre)

    print(f'{nadir} : {int(mask.sum())} points dans la grille') # compte le nombre de True dans le masque (True = 1, False = 0)

# Charger SWOT

path = '/data1/data/models/eNATL60/BLB002/obs/swot/'
files_swot = sorted(glob.glob(path + '*.nc'))
print(f'Nombre de fichiers SWOT : {len(files_swot)}')


# Boucle jour par jour

dates = [date(2010, 6, 1) + timedelta(days=i) for i in range(2)]

for d in dates:

    lats_jour = []
    lons_jour = []
    ssh_jour  = []

    # Définir la fenêtre temporelle
    t0 = np.datetime64(d)
    t1 = np.datetime64(d + timedelta(days=1))
    
    print(f'Traitement {d}...')

    for ds in ds_nadirs :
        mask_t = (ds.time >= t0) & (ds.time < t1)
        if mask_t.sum() == 0:
            continue  # pas de points ce jour, on passe au satellite suivant
        lats_jour.append(ds.lat.values[mask_t]) # contient les latitudes de tous les points observés ce jour là 
        lons_jour.append(ds.lon.values[mask_t])
        ssh_jour.append(ds.ssh.values[mask_t])

    for f in files_swot : 
          ds_swot = xr.open_dataset(f)

          if ds_swot.time.min() > t1 or ds_swot.time.max() < t0 : 
              continue # fichier hors période
          
          # Masque temporel (2D car time x num_pixels)
          mask_t = (ds_swot.time >= t0) & (ds_swot.time < t1) # shape (time,)

          # Masque géographique
          mask_geo = ((ds_swot.latitude >= lat_grid[0]) & (ds_swot.latitude <= lat_grid[-1]) &
                (ds_swot.longitude >= lon_grid[0]) & (ds_swot.longitude <= lon_grid[-1]))  # shape (time, 52)
          
          # Combiner : mask_t (time,) et mask_geo (time, 52)
          # On étend mask_t pour qu'il soit (time, 52) en utilisant np.newaxis
          mask_st = mask_t.values[:, np.newaxis] & mask_geo.values  # shape (time, 52)

          if mask_st.sum() == 0:
              continue
          lats_jour.append(ds_swot.latitude.values[mask_st]) # contient les latitudes de tous les points observés ce jour là 
          lons_jour.append(ds_swot.longitude.values[mask_st])
          ssh_jour.append(ds_swot.ssh.values[mask_st])

    if len(lats_jour) == 0:
        print(f'{d} : aucune observation')
        continue

    lats_all = np.concatenate(lats_jour)
    lons_all = np.concatenate(lons_jour)
    ssh_all = np.concatenate (ssh_jour)

    print(f'{d} : {len(lats_all)} points')


    