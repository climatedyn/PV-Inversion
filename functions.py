from prepare_PVI import prepare_PVI
from PVI import PVinversion
from basic_functions import gradm as gradient
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

    fig=plt.figure(title,figsize=[20,12])
    ax1 = plt.subplot(221,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax2 = plt.subplot(222,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax3 = plt.subplot(223,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax4 = plt.subplot(224,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                               true_scale_latitude=None,
                                                               globe=None))
    
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
    return fig

def generateXarray(ubg,vbg,Phibg,qbg,thtbg,
                   u,v,Phi,q,tht,
                   uup,vup,Phiup,qup,thtup,
                   ulow,vlow,Philow,qlow,thtlow,
                   uTlow=None,vTlow=None,PhiTlow=None,qTlow=None,thtTlow=None,
                   uPVlow=None,vPVlow=None,PhiPVlow=None,qPVlow=None,thtPVlow=None,
                   p=None,lat=None,lon=None,day=None):
    
    result = xr.Dataset({"lon"     : ("lon", lon),
                         "lat"     : ("lat", lat),
                         "pres"    : ("pres", p),
                         "pres_NB" : ("pres_NB" ,[875,125]),
                         "u_bg"    : (["pres","lat","lon"], ubg),
                         "v_bg"    : (["pres","lat","lon"], vbg),
                         "z_bg"    : (["pres","lat","lon"], Phibg),
                         "pv_bg"   : (["pres","lat","lon"], qbg),
                         "t_bg"    : (["pres_NB","lat","lon"], thtbg),
                         "u_bal"   : (["pres","lat","lon"], u),
                         "v_bal"   : (["pres","lat","lon"], v),
                         "z_bal"   : (["pres","lat","lon"], Phi),
                         "pv_bal"  : (["pres","lat","lon"], q),
                         "t_bal"   : (["pres_NB","lat","lon"], tht),
                         "u_up"    : (["pres","lat","lon"], uup),
                         "v_up"    : (["pres","lat","lon"], vup),
                         "z_up"    : (["pres","lat","lon"], Phiup),
                         "pv_up"   : (["pres","lat","lon"], qup),
                         "t_up"    : (["pres_NB","lat","lon"], thtup),
                         "u_low"   : (["pres","lat","lon"], ulow),
                         "v_low"   : (["pres","lat","lon"], vlow),
                         "z_low"   : (["pres","lat","lon"], Philow),
                         "pv_low"  : (["pres","lat","lon"], qlow),
                         "t_low"   : (["pres_NB","lat","lon"], thtlow),
                        })
    
    if uTlow is not None:
        result = result.assign({
            "u_tlow" : (["pres","lat","lon"], uTlow),
            "v_tlow" : (["pres","lat","lon"], vTlow),
            "z_tlow" : (["pres","lat","lon"], PhiTlow),
            "pv_tlow" : (["pres","lat","lon"], qTlow),
            "t_tlow"  : (["pres_NB","lat","lon"], thtTlow)
        })
    if uPVlow is not None:
        result = result.assign({
            "u_pvlow" : (["pres","lat","lon"], uPVlow),
            "v_pvlow" : (["pres","lat","lon"], vPVlow),
            "z_pvlow" : (["pres","lat","lon"], PhiPVlow),
            "pv_pvlow": (["pres","lat","lon"], qPVlow),
            "t_pvlow" : (["pres_NB","lat","lon"], thtPVlow)
        })

    # add description for the variables
    result["u_bg"].attrs["title"] = "Balanced U_velocity of background flow"
    result["u_bg"].attrs["units"] = "m s**-1"
    result["v_bg"].attrs["title"] = "Balanced V_velocity of background flow"
    result["v_bg"].attrs["units"] = "m s**-1"
    result["z_bg"].attrs["title"] = "Geopotential associated with background PV"
    result["z_bg"].attrs["units"] = "m**2 s**-2"
    result["t_bg"].attrs["title"] = "Potential temperature for BGinversion"
    result["t_bg"].attrs["units"] = "K"
    result["pv_bg"].attrs["title"] = "PV of background PV"
    result["pv_bg"].attrs["units"] = "PVU"

    result["u_bal"].attrs["title"] = "Balanced U_velocity"
    result["u_bal"].attrs["units"] = "m s**-1"
    result["v_bal"].attrs["title"] = "Balanced V_velocity"
    result["v_bal"].attrs["units"] = "m s**-1"
    result["z_bal"].attrs["title"] = "Geopotential associated with full PV"
    result["z_bal"].attrs["units"] = "m**2 s**-2"
    result["t_bal"].attrs["title"] = "Potential temperature for full inversion"
    result["t_bal"].attrs["units"] = "K"
    result["pv_bal"].attrs["title"] = "full PV"
    result["pv_bal"].attrs["units"] = "PVU"

    result["u_up"].attrs["title"] = "Balanced U_velocity associated with \
                                                        upper-level PV anomalies"
    result["u_up"].attrs["units"] = "m s**-1"
    result["v_up"].attrs["title"] = "Balanced V_velocity associated \
                                                        with upper-level PV anomalies"
    result["v_up"].attrs["units"] = "m s**-1"
    result["z_up"].attrs["title"] = "Geopotential associated with upper-level PV"
    result["z_up"].attrs["units"] = "m**2 s**-2"
    result["t_up"].attrs["title"] = "Potential temperature for UP inversion"
    result["t_up"].attrs["units"] = "K"
    result["pv_up"].attrs["title"] = "upper-level PV"
    result["pv_up"].attrs["units"] = "PVU"

    result["u_low"].attrs["title"] = "Balanced U_velocity associated with \
                                                    low-level PV anomalies"
    result["u_low"].attrs["units"] = "m s**-1"
    result["v_low"].attrs["title"] = "Balanced V_velocity associated with \
                                                    low-level PV anomalies"
    result["v_low"].attrs["units"] = "m s**-1"
    result["z_low"].attrs["title"] = "Geopotential associated with low-level PV"
    result["z_low"].attrs["units"] = "m**2 s**-2"
    result["t_low"].attrs["title"] = "Potential temperature for LOWinversion"
    result["t_low"].attrs["units"] = "K"
    result["pv_low"].attrs["title"] = "low-level PV"
    result["pv_low"].attrs["units"] = "PVU"

    # rotate the result back to the original shape and add a time coordinate
    result.coords["time"] = day
    result = result.transpose().expand_dims("time", 0)

    return result
        
    
    
def ComputeInstantInversion(data,dataBG,latlim=[25,80],lonlim=[0,359],BGinversion=False,FULLinversion=False,UPinversion=False,LOWinversion=False,TLOWinversion=False,PVLOWinversion=False):
    '''This function is closely modelled in `run_PVI.py` provided by the original package.

    main file to execute piecewise PV-inversion as defined in Teubler and Riemer 2016
    1. first calls prepare_PVI, here input-variables for PV inversion are calculated based
    on wind field, temperature and geopotential
    2. PV-inversion is called
    3. wind fields calculated from streamfunction are saved in netcdf-file

    The input data contains instantaneous data
    The input dataBG is typically calculated from a 30-day time mean
    '''
    data = ac.StandardGrid(data,rename=True)
    dataBG = ac.StandardGrid(dataBG,rename=True)
    
    
    dlatlon = np.quantile(np.diff(data['lat']),0.5)


    # ###################       no changes below        ##########################################
    latsel = {'lat':np.linspace(latlim[1],latlim[0],int((latlim[1]-latlim[0])/dlatlon+1))}

    # reduce data to lat, -lon range of interest and sort pressure levels
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
        Psi_bg,Phi_bg   = PVinversion.PVinversion(qbg, S, H, tht_bg, lat, lon, p , underrelax=0.5)[:2]

        ubg, vbg = gradient(Psi_bg,lat,lon)
        ubg = -ubg

    if FULLinversion:
        q, S, H, tht, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'full')
        Psi,Phi         = PVinversion.PVinversion(q, S, H, tht, lat, lon, p, underrelax=0.5)[:2]

        u, v = gradient(Psi,lat,lon)
        u = -u

    if UPinversion:
        qup, S, H, tht_up, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'up',
                                      dataBG.u,dataBG.v,dataBG.t,dataBG.z)
        Psi_up,Phi_up     = PVinversion.PVinversion(qup, S, H, tht_up, lat, lon, p, underrelax=0.5)[:2]

        uup, vup = gradient(Psi_up,lat,lon)
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
        Psi_low,Phi_low    = PVinversion.PVinversion(qlow, S, H, tht_low, lat, lon , p, underrelax=0.5)[:2]

        ulow, vlow = gradient(Psi_low,lat,lon)
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
        Psi_Tlow,Phi_Tlow = PVinversion.PVinversion(qTlow, S, H, tht_Tlow, lat, lon , p, underrelax=0.5)[:2]

        uTlow, vTlow = gradient(Psi_Tlow,lat,lon)
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
        Psi_PVlow,Phi_PVlow = PVinversion.PVinversion(qPVlow, S, H, tht_PVlow, lat, lon , p, underrelax=0.5)[:2]

        uPVlow, vPVlow = gradient(Psi_PVlow,lat,lon)
        uPVlow = -uPVlow

        # calculate wind field of lower anomalies due to substraction method
        uPVLOW = u - uPVlow
        vPVLOW = v - vPVlow
        PhiPVLOW = Phi-Phi_PVlow
        qPVLOW   = q-qPVlow
        thtPVLOW = tht-tht_PVlow


    PVIXR = generateXarray(ubg,vbg,Phi_bg,qbg,tht_bg,
                               u,v,Phi,q,tht,
                               uUP,vUP,PhiUP,qUP,thtUP,
                               uLOW,vLOW,PhiLOW,qLOW,thtLOW,
                               uTlow=uTLOW,vTlow=vTLOW,PhiTlow=PhiTLOW,qTlow=qTLOW,thtTlow=thtTLOW,
                               uPVlow=uPVLOW,vPVlow=vPVLOW,PhiPVlow=PhiPVLOW,qPVlow=qPVLOW,thtPVlow=thtPVLOW,
                               p=p,lat=lat,lon=lon,day=data.time)

    return PVIXR


def ComputeInversion(data,dataBG,latlim=[25,80],lonlim=[0,359],BGinversion=False,FULLinversion=False,UPinversion=False,LOWinversion=False,TLOWinversion=False,PVLOWinversion=False):
    '''Call instant inversion at every timestep. Assumes dataBG is either independent of time or has the same time dimension as data (for instance, rolling mean).
    '''
    if 'time' in data.dims and len(data.time > 1):
        has_time = True
    else:
        has_time = False
        data = data.squeeze()
    
    if 'time' in dataBG.dims and len(dataBG.time > 1):
        bg_time = True
    else:
        bg_time = False
        dataBG = dataBG.squeeze()

    if not has_time:
        return ComputeInstantInversion(data,dataBG,latlim,lonlim,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)
    else:
        dst = []
        for t,time in enumerate(data.time):
            if bg_time:
                dbg = dataBG.isel(time=t)
            else:
                dbg = dataBG
            ds = ComputeInstantInversion(data.isel(time=t),dbg,latlim,lonlim,BGinversion,FULLinversion,UPinversion,LOWinversion,TLOWinversion,PVLOWinversion)
            ds['time'] = time
            dst.append(ds)
        return xr.concat(dst,'time')

