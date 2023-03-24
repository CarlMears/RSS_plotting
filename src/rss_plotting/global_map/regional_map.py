import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset as netcdf_dataset

import typing

def plot_regional_subset_map(a, 
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
                central_latitude:float=20.0,
                longitude_size:float=60.0,
                latitude_size:float=30.0,
                units:str = '',
                return_map:bool = False,
                panel_label:str = None,
                ):

    '''Plots a regional subset of a globally defined map
       central_longitide and central latitude define the center of the map
       longitude_size and latitude_size define the extent in the lat/lon directions'''
    img_extent = [-180.15, 179.85, -90.0, 90.0]

    min_lon = central_longitude - longitude_size/2.0
    max_lon = central_longitude + longitude_size/2.0
    min_lat = central_latitude  - latitude_size/2.0
    max_lat = central_latitude  + latitude_size/2.0

    display_extent = [min_lon,max_lon, min_lat, max_lat]
    if fig_in is None:
        fig = plt.figure(figsize=(5, 6)) 
    else:
        fig = fig_in

    if ax_in is None:
        proj=ccrs.PlateCarree(central_longitude = central_longitude)
        ax = fig.add_subplot(1, 1, 1, projection=proj,title=title)
    else:
        ax = ax_in
        proj=ax.projection

    for item in ([ax.title]):
        item.set_fontsize(16)

    sz = a.shape
    num_lons = sz[1]
    cmap_copy = plt.get_cmap(cmap).copy()
    cmap_copy.set_bad('grey')

    roll_amount = int((num_lons/360.0)*(180-central_longitude))

    map = ax.imshow(np.flipud(np.roll(a, roll_amount, axis=1)), cmap=cmap_copy, origin='upper', transform=proj,
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)

    if plt_colorbar:
        cbar = plt.colorbar(map, shrink=0.7, orientation='vertical',ax=ax)
        cbar.ax.tick_params(labelsize=14)

    #coast = cfeature.GSHHSFeature(scale= 'high', levels=[1], linewidth=0.7)
    coast = cfeature.COASTLINE
    ax.add_feature(coast) 
    #lakes = cfeature.GSHHSFeature(scale= 'high', levels=[2], linewidth=0.3)
    #lakes = cfeature.LAKES
    #ax.add_feature(lakes)

    lake_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '50m',edgecolor='black',facecolor='none')
    ax.add_feature(lake_50m) 
   
   
    if title is not None:
        ax.set_title(title)
    ax.set_global()

    ax.text(0.5, -0.2, units, va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes,fontsize=14)
    
    ax.set_extent(display_extent,crs = ccrs.PlateCarree())

    if panel_label is not None:
        plt.text(panel_label_loc[0],panel_label_loc[1],panel_label,transform=ax.transAxes,fontsize=14,
                 bbox={"facecolor": 'white',"edgecolor": 'white'})

    if return_map:
        return fig, ax, map
    else:
        return fig, ax

if __name__ == '__main__':
    import xarray as xr
    import numpy as np
    from rss_plotting.global_map import plot_global_map

    test_file = 'L:/access/amsr2_daily_test/Y2021/M07/amsr2_resamp_tbs_2021_07_01.nc'
    d = xr.open_dataset(test_file)

    





    central_longitude=270
    fig = plt.figure(figsize=(8,10))
    axs = []
    for m in range(0,4):
        axs.append(fig.add_subplot(2,2,m+1,projection=ccrs.PlateCarree(central_longitude = central_longitude)))
    tpw = d['columnar_water_vapor'][:,:,8].values
    fig,axs[0] = plot_regional_subset_map(tpw,fig_in = fig,ax_in = axs[0],
                                    vmin=0.0,
                                    vmax=70.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='ERA5 Column Water Vapor (mm)',
                                    plt_colorbar=True,)
 
    
    skt = d['skt'][:,:,8].values
    fig,axs[1] = plot_regional_subset_map(skt,fig_in = fig,ax_in = axs[1],
                                    vmin=270.0,
                                    vmax=300.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='ERA5 Surface Temperature ',
                                    plt_colorbar=True,cmap='magma')
    
    cld = d['rainfall_rate'][:,:,8].values
    fig,axs[2] = plot_regional_subset_map(cld,fig_in = fig,ax_in = axs[2],
                                    vmin=0.0,
                                    vmax=10.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='IMERG rainfall rate (mm/hr)',
                                    plt_colorbar=True,cmap='magma_r')

    lf = d['land_area_fraction'].values
    fig,axs[3] = plot_regional_subset_map(lf,fig_in = fig,ax_in = axs[3],
                                    vmin=0.0,
                                    vmax=1.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='Land Fraction',
                                    plt_colorbar=True,cmap='BrBG_r')

    fig.subplots_adjust(bottom=0.1,top=0.9,left = 0.04,right=0.95,wspace=0.08,hspace=0.08)

    png_file = 'M:/job_access/plots/fig1.png'
    fig.savefig(png_file)
    plt.show()
    print

    fig2 = plt.figure(figsize=(8,10))
    axs2 = []
    tb_cmap = 'viridis'
    for m in range(0,4):
        axs2.append(fig2.add_subplot(2,2,m+1,projection=ccrs.PlateCarree(central_longitude = central_longitude)))
    
    amsr2_11V = d['brightness_temperature'][:,:,8,4]
    fig2,axs2[0] = plot_regional_subset_map(amsr2_11V,fig_in = fig2,ax_in = axs2[0],
                                    vmin=80.0,
                                    vmax=300.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 11V Tb (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    
    amsr2_11H = d['brightness_temperature'][:,:,8,5]
    fig2,axs2[1] = plot_regional_subset_map(amsr2_11H,fig_in = fig2,ax_in = axs2[1],
                                    vmin=80.0,
                                    vmax=300.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 11H Tb (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    amsr2_24V = d['brightness_temperature'][:,:,8,8]
    fig2,axs2[2] = plot_regional_subset_map(amsr2_24V,fig_in = fig2,ax_in = axs2[2],
                                    vmin=80.0,
                                    vmax=300.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 24V Tb (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    amsr2_24H = d['brightness_temperature'][:,:,8,9]
    fig2,axs2[3] = plot_regional_subset_map(amsr2_24H,fig_in = fig2,ax_in = axs2[3],
                                    vmin=80.0,
                                    vmax=300.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 24H Tb (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    fig2.subplots_adjust(bottom=0.1,top=0.9,left = 0.04,right=0.95,wspace=0.08,hspace=0.08)

    png_file = 'M:/job_access/plots/fig2.png'
    fig2.savefig(png_file)

    fig3 = plt.figure(figsize=(8,10))
    axs2 = []
    tb_cmap = 'viridis'
    for m in range(0,4):
        axs2.append(fig3.add_subplot(2,2,m+1,projection=ccrs.PlateCarree(central_longitude = central_longitude)))
    print(d['upwelling_tb'].shape)
    amsr2_11V = d['upwelling_tb'][:,:,8,1]
    fig3,axs2[0] = plot_regional_subset_map(amsr2_11V,fig_in = fig3,ax_in = axs2[0],
                                    vmin=0.0,
                                    vmax=20.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 11 Tb up (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    
    amsr2_11H = d['transmissivity'][:,:,8,1]
    fig3,axs2[1] = plot_regional_subset_map(amsr2_11H,fig_in = fig3,ax_in = axs2[1],
                                    vmin=0.8,
                                    vmax=1.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 11 Transmissivity',
                                    plt_colorbar=True,cmap=tb_cmap)

    amsr2_37V = d['upwelling_tb'][:,:,8,3]
    fig3,axs2[2] = plot_regional_subset_map(amsr2_37V,fig_in = fig3,ax_in = axs2[2],
                                    vmin=30.0,
                                    vmax=130.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 24 Tb up (K)',
                                    plt_colorbar=True,cmap=tb_cmap)

    amsr2_37H = d['transmissivity'][:,:,8,3]
    fig3,axs2[3] = plot_regional_subset_map(amsr2_37H,fig_in = fig3,ax_in = axs2[3],
                                    vmin=0.5,
                                    vmax=1.0,
                                    central_latitude=40,
                                    central_longitude=270,
                                    latitude_size=30,
                                    longitude_size=30,
                                    title='AMSR2 24 Transmissivity',
                                    plt_colorbar=True,cmap=tb_cmap)
    fig3.subplots_adjust(bottom=0.1,top=0.9,left = 0.04,right=0.95,wspace=0.08,hspace=0.08)
    png_file = 'M:/job_access/plots/fig3.png'
    fig3.savefig(png_file)
    plt.show()

    print
