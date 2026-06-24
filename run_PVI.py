from xarray import open_dataset as read
import numpy as np
from PV_Inversion import functions as fc


'''
    main file to execute piecewise PV-inversion as defined in Teubler and Riemer 2016
    1. first calls prepare_PVI, here input-variables for PV inversion are calculated based
    on wind field, temperature and geopotential
    2. PV-inversion is called
    3. wind fields calculated from streamfunction are saved in netcdf-file

    The input file for data is a standard grib-file from analysis IFS-data
    The input file for dataBG is a netcdf-file calculated from a 30-day time mean
'''

BGinversion     = True
FULLinversion   = True
UPinversion     = True
LOWinversion    = True
TLOWinversion   = True #(only low-level temperature inversion)
PVLOWinversion  = True #(only low-level PV inversion)

plot_figure   = True
save_data     = True


latlim  = [25, 80]
lonlim  = [0, 359]

data   = read('data/YOTC_20081031_18.nc')
dataBG = read('data/TM20081013_06-TM30.nc')


PVIXR = fc.ComputeInversion(data,dataBG,latlim,lonlim,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)

if plot_figure:
    fig = fc.plotting((PVIXR.v_bal,PVIXR.u_bal),(PVIXR.v_bg,PVIXR.u_bg),(PVIXR.v_up,PVIXR.u_up),(PVIXR.v_low,PVIXR.u_low),PVIXR.lat,PVIXR.lon,250,'flow of inversion')
    outFile = 'pvinv.png'
    fig.savefig(outFile)
    print(outFile)

if save_data:
    outFile = 'PVIout.nc'
    PVIXR.to_netcdf(outFile)
    print(outFile)
