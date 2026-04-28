import xarray as xr
import glob
import os

def inspecter_fichier(base_path="/data1/data/models/eNATL60/BLB002/obs"):
    files_nadir = sorted(glob.glob(os.path.join(base_path, "nadirs", "*.nc")))
    files_swot  = sorted(glob.glob(os.path.join(base_path, "swot", "*.nc")))

    print(f'Nadirs : {len(files_nadir)} fichiers')
    print(f'SWOT   : {len(files_swot)} fichiers')

    if files_nadir:
        print('\n=== NADIR ===')
        print(xr.open_dataset(files_nadir[0]))

    if files_swot:
        print('\n=== SWOT ===')
        print(xr.open_dataset(files_swot[0]))

if __name__ == "__main__":
    inspecter_fichier()