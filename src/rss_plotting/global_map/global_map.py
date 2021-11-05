import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from netCDF4 import Dataset as netcdf_dataset

import typing
import warnings

def global_map(a, 
                fig_in = None,
                ax_in = None,
                extent = None,
                panel_label_loc = [0.03,0.90],
                vmin:float=0.0, 
                vmax:float=30.0, 
                cmap:str = 'BrBG', 
                plt_colorbar:bool = False,
                title:str='',
                central_longitude:float=0.0,
                units:str = '',
                return_map:bool = False,
                panel_label:str = None,
                ):

    warnings.warn("Warning: global_map() is deprecated - use plot_global_map()")

    
    return plot_global_map(a, 
                fig_in = fig_in,
                ax_in = ax_in,
                extent = extent,
                panel_label_loc = panel_label_loc,
                vmin=vmin, 
                vmax=vmax, 
                cmap = cmap, 
                plt_colorbar = plt_colorbar,
                title=title,
                central_longitude=central_longitude,
                units = units,
                return_map = return_map,
                panel_label = panel_label,
                )

def plot_global_map(a, 
                fig_in = None,
                ax_in = None,
                extent = None,
                panel_label_loc = [0.03,0.90],
                vmin:float=0.0, 
                vmax:float=30.0, 
                cmap:str = 'BrBG', 
                plt_colorbar:bool = False,
                title:str='',
                central_longitude:float=0.0,
                units:str = '',
                return_map:bool = False,
                panel_label:str = None,
                ):


    img_extent = [-180.0, 180.0, -90.0, 90.0]
    if fig_in is None:
        fig = plt.figure(figsize=(10, 5))  # type: Figure
    else:
        fig = fig_in

    if ax_in is None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = central_longitude),title=title)
    else:
        ax = ax_in

    for item in ([ax.title]):
        item.set_fontsize(16)
    sz = a.shape
    num_lons = sz[1]
    cmap_copy = plt.get_cmap(cmap).copy()
    cmap_copy.set_bad('grey')
    map = ax.imshow(np.flipud(np.roll(a, int(num_lons/2), axis=1)), cmap=cmap_copy, origin='upper', transform=ccrs.PlateCarree(),
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)
    if plt_colorbar:
        cbar = fig.colorbar(map, shrink=0.7, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
    try:
        ax.coastlines()
    except:
        print('Trouble getting coastline file')

    ax.set_global()
    ax.text(0.5, -0.2, units, va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
    if extent is not None:
        ax.set_extent(extent,crs = ccrs.PlateCarree())

    if panel_label is not None:
        plt.text(panel_label_loc[0],panel_label_loc[1],panel_label,transform=ax.transAxes,fontsize=14,
                 bbox={"facecolor": 'white',"edgecolor": 'white'})

    if return_map:
        return fig, ax, map
    else:
        return fig, ax

def global_map_w_zonal_mean(a, 
                    vmin=0.0, 
                    vmax=30.0, 
                    zmin=None,
                    zmax=None,
                    cmap=None, 
                    plt_colorbar=False,
                    title='',
                    central_longitude = 0.0,
                    units = '',
                    plot_zonal = True):

    sz = a.shape
    num_lons = sz[1]
    num_lats = sz[0]

    if plot_zonal:
        #calculate zonal means
        zonal_mean = np.zeros((num_lats))
        for ilat in range(0,num_lats):
            row = a[ilat,:]
            zonal_mean[ilat] = np.nanmean(row)

    # definitions for the axes
    left, width = 0.03, 0.70
    bottom, height = 0.2, 0.70
    spacing = 0.1
    rect_map = [left, bottom, width, height]
    rect_zonal = [left + width + spacing, bottom, 0.14, height]
    rect_cbar = [left+0.1, bottom - 0.12, width - 0.2, 0.05]
    
    # start with a square Figure
    fig = plt.figure(figsize=(10, 5))

    ax_map   = fig.add_axes(rect_map,projection=ccrs.PlateCarree(central_longitude = central_longitude),title=title)
    ax_zonal = fig.add_axes(rect_zonal, sharey=ax_map)
    ax_cbar  = fig.add_axes(rect_cbar)

    img_extent = [-180.0, 180.0, -90.0, 90.0]
    if len(title) < 30:
        ax_map.title.set_fontsize(16)
    else:
        ax_map.title.set_fontsize(12)

    mp = ax_map.imshow(np.flipud(np.roll(a, int(num_lons/2), axis=1)), cmap=cmap, origin='upper', transform=ccrs.PlateCarree(),
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)
    if plt_colorbar:
        cbar = fig.colorbar(mp, cax=ax_cbar, shrink=0.7, orientation='horizontal')
        cbar.ax.set_title(units)
        cbar.ax.tick_params(labelsize=14)
    # try:
    #     ax_map.coastlines()
    # except:
    #     print('Trouble getting coastline file')

    ax_map.set_global()

    dlat = 180.0/num_lats
    lats = -90.0 + dlat/2.0 + dlat*np.arange(0.0,num_lats)
    ax_zonal.plot(zonal_mean,lats)
    ax_zonal.set_ylabel('Latitude')
    if (zmin is not None) and (zmax is not None):
        ax_zonal.set_xlim([zmin,zmax])
    return fig, [ax_map,ax_zonal]

def write_global_map_netcdf(data_in,file_nc='',standard_name='',long_name = '',valid_range = [],nc_data_type='f4'):

    
    shp = data_in.shape
    lats = np.arange(0,720)*0.25 - 90.0 + 0.125
    lons = np.arange(0,1440)*0.25 + 0.125

    if len(shp) != 2:
        raise ValueError('data should be a 2 dimensional array')

    root_grp = netcdf_dataset(file_nc, 'w', format='NETCDF4')

    root_grp.createDimension('latitude', 720)
    root_grp.createDimension('longitude', 1440)

    latitude = root_grp.createVariable('latitude', 'f4', ('latitude',))
    longitude = root_grp.createVariable('longitude', 'f4', ('longitude',))
    data = root_grp.createVariable(standard_name, nc_data_type, ('latitude', 'longitude',))
   


    latitude.standard_name = "latitude"
    latitude.long_name = "latitude"
    latitude.units = "degrees_north"
    latitude.valid_range = (-90.0, 90.0)
    latitude.FillValue = -999.0
    latitude.renameAttribute('FillValue', '_FillValue')

    longitude.standard_name = "longitude"
    longitude.long_name = "longitude"
    longitude.units = "degrees_east"
    longitude.valid_range = (0.0, 360.0)
    longitude.FillValue = -999.0
    longitude.renameAttribute('FillValue', '_FillValue')

    data.standard_name = standard_name
    data.long_name = long_name

    latitude[:] = lats
    longitude[:] = lons
    data[:, :] = data_in
    root_grp.close()

    print('Wrote: ',file_nc)

    return