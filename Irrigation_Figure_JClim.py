import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


Irrigation_data = '/Users/gbromley/Downloads/10_13019_M20599/HID_v10/HID_aei_ha.nc'
Irrig = xr.open_dataset(Irrigation_data)
AEI = Irrig['AEI_ha']

ext_e = -92
ext_w = -120
ext_n = 55
ext_s = 38

AEI = AEI.sel(lat=slice(ext_s-5,ext_n+5),lon=slice(ext_w-5,ext_e+5))


AEI.isel(prod_time=50).plot()
plt.show()

AEI