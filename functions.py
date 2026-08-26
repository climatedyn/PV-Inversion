from .prepare_PVI import prepare_PVI
from .PVinversion import PVinversion
from .basic_functions import gradm as gradient
import xarray as xr
import numpy as np
from aostools import climate as ac


def plotting(uv,uvBG,uvNT,uvTD,lat,lon,p,title):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    #from basics.plot.mapping import stereo

    uv = (uv[0].sel(pres=p).squeeze().transpose('lat','lon'),uv[1].sel(pres=p).squeeze().transpose('lat','lon'))
    uvBG=(uvBG[0].sel(pres=p).squeeze().transpose('lat','lon'),uvBG[1].sel(pres=p).squeeze().transpose('lat','lon'))
    uvNT=(uvNT[0].sel(pres=p).squeeze().transpose('lat','lon'),uvNT[1].sel(pres=p).squeeze().transpose('lat','lon'))
    uvTD=(uvTD[0].sel(pres=p).squeeze().transpose('lat','lon'),uvTD[1].sel(pres=p).squeeze().transpose('lat','lon'))

    llon,llat = np.meshgrid(lon,lat)

    if np.mean(lat) > 0:
        proj = ccrs.NorthPolarStereo(central_longitude=0,true_scale_latitude=None,globe=None)
    else:
        proj = ccrs.SouthPolarStereo(central_longitude=0,true_scale_latitude=None,globe=None)
        

    fig=plt.figure(title,figsize=[20,12])
    ax1 = plt.subplot(221,
                      projection=proj)
    ax2 = plt.subplot(222,
                      projection=proj)
    ax3 = plt.subplot(223,
                      projection=proj)
    ax4 = plt.subplot(224,
                      projection=proj)
    
    #stereo(ax1,(min(lat),max(lat)))
    #stereo(ax2,(min(lat),max(lat)))
    #stereo(ax3,(min(lat),max(lat)))
    #stereo(ax4,(min(lat),max(lat)))

    # ------------------------------------------------------------------------------
    # ##############################################################################
    # ------------------------------------------------------------------------------

    CS1 = ax1.contourf(lon,lat,np.hypot(uv[0],uv[1]).squeeze(),
                      np.linspace(0,80,int((80/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax1.quiver(llon[::3,::3],llat[::3,::3],
               uv[1][::3,::3].squeeze(),uv[0][::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS1,ax=ax1, orientation='vertical',aspect=20,fraction=0.04)
    ax1.set_title('balanced flow (after inversion)')

    # ------------------------------------------------------------------------------
    CS2 = ax2.contourf(lon,lat,np.hypot(uvNT[0],uvNT[1]).squeeze(),
                      np.linspace(0,60,int((60/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax2.quiver(llon[::3,::3],llat[::3,::3],
               uvNT[1][::3,::3].squeeze(),uvNT[0][::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS2,ax=ax2, orientation='vertical',aspect=20,fraction=0.04)
    ax2.set_title('balanced flow of upper-level inversion')
    # ------------------------------------------------------------------------------
    CS3 = ax3.contourf(lon,lat,np.hypot(uvTD[0],uvTD[1]).squeeze(),
                      np.linspace(0,10,int((10/1)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax3.quiver(llon[::3,::3],llat[::3,::3],
               uvTD[1][::3,::3].squeeze(),uvTD[0][::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS3,ax=ax3, orientation='vertical',aspect=20,fraction=0.04)
    ax3.set_title('balanced flow of low-level inversion')
    # ------------------------------------------------------------------------------
    CS4 = ax4.contourf(lon,lat,np.hypot(uvBG[0],uvBG[1]).squeeze(),
                      np.linspace(0,60,int((60/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax4.quiver(llon[::3,::3],llat[::3,::3],
               uvBG[1][::3,::3].squeeze(),uvBG[0][::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS4,ax=ax4, orientation='vertical',aspect=20,fraction=0.04)
    ax4.set_title('balanced background flow (after inversion)')

    for ax in [ax1,ax2,ax3,ax4]:
        ax.coastlines()
    
    return fig

def generateXarray(ubg,vbg,Phibg,Psibg,qbg,thtbg,
                   u,v,Phi,Psi,q,tht,
                   uup,vup,Phiup,Psiup,qup,thtup,
                   ulow,vlow,Philow,Psilow,qlow,thtlow,
                   uTlow=None,vTlow=None,PhiTlow=None,PsiTlow=None,qTlow=None,thtTlow=None,
                   uPVlow=None,vPVlow=None,PhiPVlow=None,PsiPVlow=None,qPVlow=None,thtPVlow=None,
                   p=None,lat=None,lon=None,day=None):

    result_dict = {"lon"     : ("lon", lon),
                   "lat"     : ("lat", lat),
                   "pres"    : ("pres", p),
                   "pres_NB" : ("pres_NB" ,[875,125])}
    if ubg is not None:
        result_dict = result_dict | {
                         "u_bg"    : (["pres","lat","lon"], ubg),
                         "v_bg"    : (["pres","lat","lon"], vbg),
                         "z_bg"    : (["pres","lat","lon"], Phibg),
                         "psi_bg"  : (["pres","lat","lon"], Psibg),
                         "pv_bg"   : (["pres","lat","lon"], qbg),
                         "t_bg"    : (["pres_NB","lat","lon"], thtbg)
            }
    if u is not None:
        result_dict = result_dict | {
                         "u_bal"   : (["pres","lat","lon"], u),
                         "v_bal"   : (["pres","lat","lon"], v),
                         "z_bal"   : (["pres","lat","lon"], Phi),
                         "psi_bal" : (["pres","lat","lon"], Psi),
                         "pv_bal"  : (["pres","lat","lon"], q),
                         "t_bal"   : (["pres_NB","lat","lon"], tht)
            }
    if uup is not None:
        result_dict = result_dict | {
                         "u_up"    : (["pres","lat","lon"], uup),
                         "v_up"    : (["pres","lat","lon"], vup),
                         "z_up"    : (["pres","lat","lon"], Phiup),
                         "psi_up"  : (["pres","lat","lon"], Psiup),
                         "pv_up"   : (["pres","lat","lon"], qup),
                         "t_up"    : (["pres_NB","lat","lon"], thtup)
            }
    if ulow is not None:
        result_dict = result_dict | {
                         "u_low"   : (["pres","lat","lon"], ulow),
                         "v_low"   : (["pres","lat","lon"], vlow),
                         "z_low"   : (["pres","lat","lon"], Philow),
                         "psi_low" : (["pres","lat","lon"], Psilow),
                         "pv_low"  : (["pres","lat","lon"], qlow),
                         "t_low"   : (["pres_NB","lat","lon"], thtlow)
            }
    result = xr.Dataset(result_dict)
    
    if uTlow is not None:
        result = result.assign({
            "u_tlow" : (["pres","lat","lon"], uTlow),
            "v_tlow" : (["pres","lat","lon"], vTlow),
            "z_tlow" : (["pres","lat","lon"], PhiTlow),
            "psi_tlow": (["pres","lat","lon"], PsiTlow),
            "pv_tlow" : (["pres","lat","lon"], qTlow),
            "t_tlow"  : (["pres_NB","lat","lon"], thtTlow)
        })
    if uPVlow is not None:
        result = result.assign({
            "u_pvlow" : (["pres","lat","lon"], uPVlow),
            "v_pvlow" : (["pres","lat","lon"], vPVlow),
            "z_pvlow" : (["pres","lat","lon"], PhiPVlow),
            "psi_pvlow": (["pres","lat","lon"], PsiPVlow),
            "pv_pvlow": (["pres","lat","lon"], qPVlow),
            "t_pvlow" : (["pres_NB","lat","lon"], thtPVlow)
        })

    # add description for the variables
    if ubg is not None:
        result["u_bg"].attrs["title"] = "Balanced U_velocity of background flow"
        result["u_bg"].attrs["units"] = "m s**-1"
        result["v_bg"].attrs["title"] = "Balanced V_velocity of background flow"
        result["v_bg"].attrs["units"] = "m s**-1"
        result["z_bg"].attrs["title"] = "Geopotential associated with background PV"
        result["z_bg"].attrs["units"] = "m**2 s**-2"
        result["psi_bg"].attrs["title"] = "Streamfunction associated with background PV"
        result["psi_bg"].attrs["units"] = "m**2 s**-1"
        result["t_bg"].attrs["title"] = "Potential temperature for BGinversion"
        result["t_bg"].attrs["units"] = "K"
        result["pv_bg"].attrs["title"] = "PV of background PV"
        result["pv_bg"].attrs["units"] = "PVU"
    if u is not None:
        result["u_bal"].attrs["title"] = "Balanced U_velocity"
        result["u_bal"].attrs["units"] = "m s**-1"
        result["v_bal"].attrs["title"] = "Balanced V_velocity"
        result["v_bal"].attrs["units"] = "m s**-1"
        result["z_bal"].attrs["title"] = "Geopotential associated with full PV"
        result["z_bal"].attrs["units"] = "m**2 s**-2"
        result["psi_bal"].attrs["title"] = "Streamfunction associated with full PV"
        result["psi_bal"].attrs["units"] = "m**2 s**-1"
        result["t_bal"].attrs["title"] = "Potential temperature for full inversion"
        result["t_bal"].attrs["units"] = "K"
        result["pv_bal"].attrs["title"] = "full PV"
        result["pv_bal"].attrs["units"] = "PVU"
    if uup is not None:
        result["u_up"].attrs["title"] = "Balanced U_velocity associated with \
                                                            upper-level PV anomalies"
        result["u_up"].attrs["units"] = "m s**-1"
        result["v_up"].attrs["title"] = "Balanced V_velocity associated \
                                                            with upper-level PV anomalies"
        result["v_up"].attrs["units"] = "m s**-1"
        result["z_up"].attrs["title"] = "Geopotential associated with upper-level PV"
        result["z_up"].attrs["units"] = "m**2 s**-2"
        result["psi_up"].attrs["title"] = "Streamfunction associated with upper-level PV"
        result["psi_up"].attrs["units"] = "m**2 s**-1"
        result["t_up"].attrs["title"] = "Potential temperature for UP inversion"
        result["t_up"].attrs["units"] = "K"
        result["pv_up"].attrs["title"] = "upper-level PV"
        result["pv_up"].attrs["units"] = "PVU"
    if ulow is not None:
        result["u_low"].attrs["title"] = "Balanced U_velocity associated with \
                                                        low-level PV anomalies"
        result["u_low"].attrs["units"] = "m s**-1"
        result["v_low"].attrs["title"] = "Balanced V_velocity associated with \
                                                        low-level PV anomalies"
        result["v_low"].attrs["units"] = "m s**-1"
        result["z_low"].attrs["title"] = "Geopotential associated with low-level PV"
        result["z_low"].attrs["units"] = "m**2 s**-2"
        result["psi_low"].attrs["title"] = "Streamfunction associated with low-level PV"
        result["psi_low"].attrs["units"] = "m**2 s**-1"
        result["t_low"].attrs["title"] = "Potential temperature for LOWinversion"
        result["t_low"].attrs["units"] = "K"
        result["pv_low"].attrs["title"] = "low-level PV"
        result["pv_low"].attrs["units"] = "PVU"

    # rotate the result back to the original shape and add a time coordinate
    result = result.transpose()

    return result
        
    
    
def ComputeInstantInversion(data,dataBG,BGinversion=False,FULLinversion=True,UPinversion=False,LOWinversion=False,TLOWinversion=False,PVLOWinversion=False):
    '''This function is closely modelled in `run_PVI.py` provided by the original package.

    main file to execute piecewise PV-inversion as defined in Teubler and Riemer 2016
    1. first calls prepare_PVI, here input-variables for PV inversion are calculated based
    on wind field, temperature and geopotential
    2. PV-inversion is called
    3. wind fields calculated from streamfunction are saved in netcdf-file

    The input data contains instantaneous data
    The input dataBG is typically calculated from a 30-day time mean
    '''
    # numpy throws an error about using 'where' without 'out'
    import warnings
    warnings.filterwarnings("ignore",category=UserWarning)
    
    data = ac.StandardGrid(data,rename=True)
    dataBG = ac.StandardGrid(dataBG,rename=True)
    

    if data['lat'].mean() < 0:
        lat_invert = True
        data['lat'] = -data['lat']
        data['v']   = -data['v']
        dataBG['lat'] = -dataBG['lat']
        dataBG['v']   = -dataBG['v']
    else:
        lat_invert = False

    data = data.sortby('lat')
    
    dlatlon = np.quantile(np.diff(data['lat']),0.5)

    latlim = [data.lat.min().values,data.lat.max().values]
    lonlim = [data.lon.min().values,data.lon.max().values]


    # ###################       no changes below        ##########################################
    latsel = {'lat':np.linspace(latlim[1],latlim[0],int((latlim[1]-latlim[0])/dlatlon+1))}

    # reduce data to lat, lon range of interest and sort pressure levels
    data   = data.sel( latsel ).squeeze()
    data   = data.sortby('pres',ascending=False)
    dataBG = dataBG.sel( latsel ).squeeze()
    dataBG = dataBG.sortby('pres',ascending=False)

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

        ubg, vbg = gradient(Psi_bg,lat,lon)
        ubg = -ubg
    else:
        ubg, vbg, Phi_bg, Psi_bg, qbg, tht_bg = None, None, None, None, None, None

    # full inversion is the basis for all partial inversions
    if FULLinversion or UPinversion or LOWinversion or TLOWinversion or PVLOWinversion:
        q, S, H, tht, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'full')
        Psi,Phi         = PVinversion(q, S, H, tht, lat, lon, p, underrelax=0.5)[:2]

        u, v = gradient(Psi,lat,lon)
        u = -u
    else:
        u, v, Phi, Psi, q, tht = None, None, None, None, None, None

    if UPinversion:
        qup, S, H, tht_up, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'up',
                                      dataBG.u,dataBG.v,dataBG.t,dataBG.z)
        Psi_up,Phi_up     = PVinversion(qup, S, H, tht_up, lat, lon, p, underrelax=0.5)[:2]

        uup, vup = gradient(Psi_up,lat,lon)
        uup = -uup

        # calculate wind field of upper anomalies due to substraction method
        uUP = u - uup
        vUP = v - vup
        PhiUP = Phi-Phi_up
        PsiUP = Psi-Psi_up
        qUP   = q-qup
        thtUP = tht-tht_up
    else:
        uUP, vUP, PhiUP, PsiUP, qUP, thtUP = None, None, None, None, None, None


    if LOWinversion:
        qlow, S, H, tht_low, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'low',
                                      dataBG.u,dataBG.v,dataBG.t,dataBG.z)
        Psi_low,Phi_low    = PVinversion(qlow, S, H, tht_low, lat, lon , p, underrelax=0.5)[:2]

        ulow, vlow = gradient(Psi_low,lat,lon)
        ulow = -ulow

        # calculate wind field of lower anomalies due to substraction method
        uLOW = u - ulow
        vLOW = v - vlow
        PhiLOW = Phi-Phi_low
        PsiLOW = Psi-Psi_low
        qLOW   = q-qlow
        thtLOW = tht-tht_low
    else:
        uLOW, vLOW, PhiLOW, PsiLOW, qLOW, thtLOW = None, None, None, None, None, None

    if TLOWinversion:
        qTlow, S, H, tht_Tlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'Tlow',
                                      dataBG.u,dataBG.v,dataBG.t,dataBG.z)
        Psi_Tlow,Phi_Tlow = PVinversion(qTlow, S, H, tht_Tlow, lat, lon , p, underrelax=0.5)[:2]

        uTlow, vTlow = gradient(Psi_Tlow,lat,lon)
        uTlow = -uTlow

        # calculate wind field of lower anomalies due to substraction method
        uTLOW = u - uTlow
        vTLOW = v - vTlow
        PhiTLOW = Phi-Phi_Tlow
        PsiTLOW = Psi-Psi_Tlow
        qTLOW   = q-qTlow
        thtTLOW = tht-tht_Tlow
    else:
        uTLOW, vTLOW, PhiTLOW, PsiTLOW, qTLOW, thtTLOW = None, None, None, None, None, None

    if PVLOWinversion:
        qPVlow, S, H, tht_PVlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'PVlow',
                                      dataBG.u,dataBG.v,dataBG.t,dataBG.z)
        Psi_PVlow,Phi_PVlow = PVinversion(qPVlow, S, H, tht_PVlow, lat, lon , p, underrelax=0.5)[:2]

        uPVlow, vPVlow = gradient(Psi_PVlow,lat,lon)
        uPVlow = -uPVlow

        # calculate wind field of lower anomalies due to substraction method
        uPVLOW = u - uPVlow
        vPVLOW = v - vPVlow
        PhiPVLOW = Phi-Phi_PVlow
        PsiPVLOW = Psi-Psi_PVlow
        qPVLOW   = q-qPVlow
        thtPVLOW = tht-tht_PVlow
    else:
        uPVLOW, vPVLOW, PhiPVLOW, PsiPVLOW, qPVLOW, thtPVLOW = None, None, None, None, None, None


    PVIXR = generateXarray(ubg,vbg,Phi_bg,Psi_bg,qbg,tht_bg,
                               u,v,Phi,Psi,q,tht,
                               uUP,vUP,PhiUP,PsiUP,qUP,thtUP,
                               uLOW,vLOW,PhiLOW,PsiLOW,qLOW,thtLOW,
                               uTlow=uTLOW,vTlow=vTLOW,PhiTlow=PhiTLOW,PsiTlow=PsiTLOW,qTlow=qTLOW,thtTlow=thtTLOW,
                               uPVlow=uPVLOW,vPVlow=vPVLOW,PhiPVlow=PhiPVLOW,PsiPVlow=PsiPVLOW,qPVlow=qPVLOW,thtPVlow=thtPVLOW,
                               p=p,lat=lat,lon=lon)
    
    if lat_invert:
        PVIXR['lat'] = -PVIXR['lat']
        for dvar in PVIXR.data_vars:
            if dvar[:2] == 'v_':
                PVIXR[dvar] = -PVIXR[dvar]

    return PVIXR.sortby('lon').sortby('lat')


def ComputeInversion(data,dataBG,BGinversion=False,FULLinversion=False,UPinversion=False,LOWinversion=False,TLOWinversion=False,PVLOWinversion=False):
    '''Compute PV inversion over multiple dimensions. This is typically 3D data along a time dimension, but it can also be a separation into seasons, lags, etc.
    '''

    data   = ac.StandardGrid(data  ,rename=True)
    dataBG = ac.StandardGrid(dataBG,rename=True)

    data = data.squeeze()
    dataBG = dataBG.squeeze()
    
    stacked = [dim for dim in data.dims if dim not in ['lon','lat','pres']]
    stackedBG = [dim for dim in dataBG.dims if dim not in ['lon','lat','pres']]
    common  = [s for s in stacked if s in stackedBG]
    stacked = [s for s in stacked if s not in stackedBG]

    # both data and dataBG are only functions of 3D space
    if len(stacked)==0 and len(common)==0:
        return ComputeInstantInversion(data,dataBG,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)
    # dataBG is function of 3D space, data has additional dimensions
    elif len(stacked)>0 and len(common)==0:
        dst = []
        data_stacked = data.stack(stacked=stacked)
        nstacks = len(data_stacked.stacked)
        for t in range(nstacks):
            ac.update_progress(t/nstacks)
            d_tmp = data_stacked.isel(stacked=t)
            ds = ComputeInstantInversion(d_tmp,dataBG,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)
            for coord in stacked:
                ds.coords[coord] = d_tmp[coord]
            ds.coords['stacked'] = d_tmp['stacked']
            dst.append(ds)
        ac.update_progress(1)
        ds = xr.concat(dst,'stacked').set_index(stacked=stacked)
        if len(stacked) == 1:
            return ds.rename({'stacked':stacked[0]})
        else:
            return ds.unstack({'stacked':stacked})
    # both dataBG and data have additional dimensions
    else:
        data_stacked = data.stack(stacked=stacked)
        data_stacked_common = data_stacked.stack(common=common)
        dataBG_common = dataBG.stack(common=common)
        nstacks = len(data_stacked.stacked)
        ncommon = len(dataBG_common.common)
        count=0
        dct = []
        for c in range(ncommon):
            datBG = dataBG_common.isel(common=c)
            dst = []
            for t in range(nstacks):
                ac.update_progress(count/nstacks/ncommon)
                d_tmp = data_stacked_common.isel(common=c,stacked=t)
                ds = ComputeInstantInversion(d_tmp,datBG,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)
                for coord in stacked:
                    ds.coords[coord] = d_tmp[coord]
                ds.coords['stacked'] = d_tmp['stacked']
                dst.append(ds)
                count += 1
            dc = xr.concat(dst,'stacked').set_index(stacked=stacked).unstack({'stacked':stacked})
            for coord in common:
                dc.coords[coord] = datBG[coord]
            dc.coords['common'] = datBG['common']
            dct.append(dc)
        ac.update_progress(1)
        ds = xr.concat(dct,'common').set_index(common=common)
        if len(common) == 1:
            ds = ds.rename({'common':common[0]})
        else:
            ds = ds.unstack({'common':common})
        return ds
                

