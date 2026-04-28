import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import date
import numpy as np

class eNATL60_Diffusion_Training_Dataset(Dataset):
    """
    Dataset d'entraînement pour le modèle de diffusion sur eNATL60, SSH uniquement.
    Tire des patches 128x128 aléatoires dans l'espace et le temps.
    """
    def __init__(self,
                 data_dir,
                 n_lon,
                 n_lat,
                 date_range,
                 var_stds,
                 model_file='enatl60_pre_processed.nc',
                 mean_file='enatl60_means.nc',
                 lon_buffers=[None, None],
                 lat_buffers=[None, None],
                ):
        self.data_dir = data_dir
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.date_range = date_range
        self.var_stds = var_stds
        self.lon_buffers = lon_buffers
        self.lat_buffers = lat_buffers
        self.variables = ['zos']
        self.n_channels = 1

        # Charger les données
        self.ds_model = xr.open_dataset(data_dir + model_file).astype('float32')
        ds_mean = xr.open_dataset(data_dir + mean_file).astype('float32')

        # Normalisation SSH : soustraction de la moyenne temporelle uniquement
        self.ds_model['zos'] = self.ds_model['zos'] - ds_mean['zos']

        # Appliquer les buffers spatiaux
        i_lon_min = self.lon_buffers[0]
        i_lon_max = -self.lon_buffers[1] if self.lon_buffers[1] is not None else None
        i_lat_min = self.lat_buffers[0]
        i_lat_max = -self.lat_buffers[1] if self.lat_buffers[1] is not None else None

        self.ds_model = self.ds_model.isel(
            lon=slice(i_lon_min, i_lon_max),
            lat=slice(i_lat_min, i_lat_max)
        )

        # Sélection temporelle
        self.ds_model = self.ds_model.sel(
            time=slice(str(date_range[0]), str(date_range[1])),
            drop=True
        )

        self.N_lon = self.ds_model.sizes['lon']
        self.N_lat = self.ds_model.sizes['lat']
        self.N_time = self.ds_model.sizes['time']
        print(f'Dataset prêt : {self.N_time} pas de temps, {self.N_lat}x{self.N_lon} pixels')

    def __len__(self):
        return int(1e9)

    def __getitem__(self, idx):
        still_checking = True
        while still_checking:
            # Crop aléatoire spatio-temporel
            lat_start = np.random.randint(self.N_lat - self.n_lat + 1)
            lon_start = np.random.randint(self.N_lon - self.n_lon + 1)
            t_idx = np.random.randint(self.N_time)

            indexer = {
                'lat': slice(lat_start, lat_start + self.n_lat),
                'lon': slice(lon_start, lon_start + self.n_lon),
                'time': t_idx
            }

            data = self.ds_model.isel(indexer, drop=True)

            outvar = np.zeros((1, self.n_lat, self.n_lon), dtype=np.float32)
            outvar[0,] = data['zos'].values / self.var_stds['zos']

            # Rejeter si NaN
            if not np.isnan(outvar).any():
                still_checking = False

        outvar = torch.from_numpy(outvar)
        outvar = torch.nan_to_num(outvar, nan=0.0)
        return outvar


class eNATL60_Inference_Dataset(Dataset):
    """
    Dataset d'inférence sur eNATL60, SSH uniquement.
    Parcourt séquentiellement les pas de temps sur une région fixe.
    """
    def __init__(self,
                 data_dir,
                 date_range,
                 var_stds,
                 model_file='enatl60_pre_processed.nc',
                 mean_file='enatl60_means.nc',
                ):
        self.data_dir = data_dir
        self.date_range = date_range
        self.var_stds = var_stds
        self.variables = ['zos']
        self.n_channels = 1

        self.ds_model = xr.open_dataset(data_dir + model_file).astype('float32')
        ds_mean = xr.open_dataset(data_dir + mean_file).astype('float32')

        # Normalisation
        self.ds_model['zos'] = self.ds_model['zos'] - ds_mean['zos']

        # Sélection temporelle
        self.ds_model = self.ds_model.sel(
            time=slice(str(date_range[0]), str(date_range[1])),
            drop=True
        )

        self.N_lon = self.ds_model.sizes['lon']
        self.N_lat = self.ds_model.sizes['lat']
        self.N_time = self.ds_model.sizes['time']
        print(f'Inference dataset : {self.N_time} pas de temps, {self.N_lat}x{self.N_lon} pixels')

    def __len__(self):
        return self.N_time

    def __getitem__(self, idx):
        data = self.ds_model.isel(time=idx)
        outvar = np.zeros((1, self.N_lat, self.N_lon), dtype=np.float32)
        outvar[0,] = data['zos'].values / self.var_stds['zos']
        outvar = torch.from_numpy(outvar)
        outvar = torch.nan_to_num(outvar, nan=0.0)
        return outvar
