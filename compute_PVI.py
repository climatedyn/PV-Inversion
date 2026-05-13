from PVI.prepare_PVI import prepare_PVI
from PVI.PVinversion import PVinversion
from PVI import basic_functions as bf
import xarray as xr
import numpy as np
from aostools import climate as ac

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


latlim  = [-80,-25]
lonlim  = [0, 359]
dlatlon = 1

data_root = '../era5_regrid/sh/ERA5_dm.1979'
bg_root = 'sh/ERA5_dm.clim'

#data_root = 'era5_inst'
#bg_root   = 'era5_jan'

dat = []
datBG = []
for var in ['u','v','t','z']:
    tmp = xr.open_dataset(f'{data_root}.{var}.nc')[var].isel(time=0).squeeze()
    dat.append(tmp)
    tmp = xr.open_dataset(f'{bg_root}.{var}.nc')[var].isel(time=0).squeeze()
    datBG.append(tmp)
data = xr.merge(dat)
dataBG = xr.merge(datBG)


data  = ac.StandardGrid(data  ,rename=True)
dataBG= ac.StandardGrid(dataBG,rename=True)

# code is written for NH, so flip hemisphere if SH
if max(latlim) < 0:
    data['lat'] = -data['lat']
    dataBG['lat']=-dataBG['lat']
    latlim = [-l for l in latlim[::-1]]
    invert_lat = True
else:
    invert_lat = False

# ###################       no changes below        ##########################################
latsel = {'lat':np.linspace(latlim[1],latlim[0],int((latlim[1]-latlim[0])/dlatlon+1))}

# reduce data to lat, -lon range of interest and sort pressure levels
data   = data.sortby('lat')
data   = data.sel( latsel ).squeeze()
data   = data.sortby('pres',ascending=False)
dataBG = dataBG.sortby('lat')
dataBG = dataBG.sel( latsel ).squeeze()
dataBG = dataBG.sortby('pres',ascending=False)
##data.sel(lon=np.linspace(230,450,450-230+1)%360)

lon  = np.asarray(data['lon'])
lat  = np.asarray(data['lat'])
p0   = np.asarray(data['pres'])

# check for consistency between BG and daily files regarding pressure level
assert all(np.asarray(dataBG['pres']) == p0), 'different vertical levels in BG and daily files'

if BGinversion:
    uBG = np.asarray(dataBG.u)
    vBG = np.asarray(dataBG.v)
    TBG = np.asarray(dataBG.t)
    zBG = np.asarray(dataBG.z)

    qbg, S, H, tht_bg, p = prepare_PVI(uBG,vBG,TBG,zBG,{'p':p0,'lon':lon,'lat':lat},'full')
    Psi_bg,Phi_bg   = PVinversion(qbg, S, H, tht_bg, lat, lon, p , underrelax=0.5)[:2]

    ubg, vbg = bf.gradm(Psi_bg,lat,lon)
    ubg = -ubg

if FULLinversion:
    q, S, H, tht, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'full')
    Psi,Phi         = PVinversion(q, S, H, tht, lat, lon, p, underrelax=0.5)[:2]

    u, v = bf.gradm(Psi,lat,lon)
    u = -u

if UPinversion:
    qup, S, H, tht_up, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'up',
                                       dataBG.u,dataBG.v,dataBG.t,dataBG.z)
    Psi_up,Phi_up     = PVinversion(qup, S, H, tht_up, lat, lon, p, underrelax=0.5)[:2]

    uup, vup = bf.gradm(Psi_up,lat,lon)
    uup = -uup

    # calculate wind field of upper anomalies due to substraction method
    uUP = u - uup
    vUP = v - vup
    PhiUP = Phi-Phi_up
    qUP   = q-qup
    thtUP = tht-tht_up


if LOWinversion:
    qlow, S, H, tht_low, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'low',
                                         dataBG.u,dataBG.v,dataBG.t,dataBG.z)
    Psi_low,Phi_low    = PVinversion(qlow, S, H, tht_low, lat, lon , p, underrelax=0.5)[:2]

    ulow, vlow = bf.gradm(Psi_low,lat,lon)
    ulow = -ulow

    # calculate wind field of lower anomalies due to substraction method
    uLOW = u - ulow
    vLOW = v - vlow
    PhiLOW = Phi-Phi_low
    qLOW   = q-qlow
    thtLOW = tht-tht_low

if TLOWinversion:
    qTlow, S, H, tht_Tlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'Tlow',
                                           dataBG.u,dataBG.v,dataBG.t,dataBG.z)
    Psi_Tlow,Phi_Tlow = PVinversion(qTlow, S, H, tht_Tlow, lat, lon , p, underrelax=0.5)[:2]

    uTlow, vTlow = bf.gradm(Psi_Tlow,lat,lon)
    uTlow = -uTlow

    # calculate wind field of lower anomalies due to substraction method
    uTLOW = u - uTlow
    vTLOW = v - vTlow
    PhiTLOW = Phi-Phi_Tlow
    qTLOW   = q-qTlow
    thtTLOW = tht-tht_Tlow

if PVLOWinversion:
    qPVlow, S, H, tht_PVlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'PVlow',
                                             dataBG.u,dataBG.v,dataBG.t,dataBG.z)
    Psi_PVlow,Phi_PVlow = PVinversion(qPVlow, S, H, tht_PVlow, lat, lon , p, underrelax=0.5)[:2]

    uPVlow, vPVlow = bf.gradm(Psi_PVlow,lat,lon)
    uPVlow = -uPVlow

    # calculate wind field of lower anomalies due to substraction method
    uPVLOW = u - uPVlow
    vPVLOW = v - vPVlow
    PhiPVLOW = Phi-Phi_PVlow
    qPVLOW   = q-qPVlow
    thtPVLOW = tht-tht_PVlow

data.close()
dataBG.close()

if save_data:
    PVIXR = bf.generateXarray(ubg,vbg,Phi_bg,qbg,tht_bg,
                           u,v,Phi,q,tht,
                           uUP,vUP,PhiUP,qUP,thtUP,
                           uLOW,vLOW,PhiLOW,qLOW,thtLOW,
                           uTlow=uTLOW,vTlow=vTLOW,PhiTlow=PhiTLOW,qTlow=qTLOW,thtTlow=thtTLOW,
                           uPVlow=uPVLOW,vPVlow=vPVLOW,PhiPVlow=PhiPVLOW,qPVlow=qPVLOW,thtPVlow=thtPVLOW,
                           p=p,lat=lat,lon=lon,day=data.time)

    if invert_lat:
        PVIXR['lat'] = -PVIXR['lat']
        PVIXR = PVIXR.sortby('lat')
    PVIXR.to_netcdf('data/PVIout.nc')



if plot_figure:
    PVIXR = PVIXR.squeeze().transpose('pres','lat','lon',...)
    bal = (PVIXR.V_velocity_of_BAL.values,PVIXR.U_velocity_of_BAL.values)
    bg  = (PVIXR.V_velocity_of_BG.values ,PVIXR.U_velocity_of_BG.values )
    up  = (PVIXR.V_velocity_of_UP.values ,PVIXR.U_velocity_of_UP.values )
    low = (PVIXR.V_velocity_of_LOW.values,PVIXR.U_velocity_of_LOW.values)
    bf.plotting(bal,bg,up,low,PVIXR.lat,PVIXR.lon,PVIXR.pres,'flow of inversion')
