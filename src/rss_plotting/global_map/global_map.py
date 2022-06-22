import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import cartopy.crs as ccrs
from netCDF4 import Dataset as netcdf_dataset

import typing
import warnings

def wind_cm():
    import matplotlib.colors as mcolors

    colors1 = plt.cm.RdYlBu_r(np.linspace(0.0, 1, 170))
    colors2 = plt.cm.gist_heat_r(np.linspace(0.6, 1, 86))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, v1=None, v2=None, clip=False):
        self.v1 = v1
        self.v2 = v2
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.v1, self.v2, self.vmax], [0.0, 0.333, 0.666, 1.0]
        z = np.ma.masked_array(np.interp(value, x, y), mask=np.isnan(value))

        return z

def swath_map(  a,      
                lats,
                lons,
                fig_in = None,
                ax_in = None,
                extent = None,
                panel_label_loc = [0.03,0.90],
                vmin:float=0.0, 
                vmax:float=30.0, 
                cmap:str = 'BrBG', 
                discrete_cmap = False,
                num_levels = 10,
                norm=None,
                boundaries = None,
                plt_colorbar:bool = False,
                plt_coastlines:bool = True,
                title:str='',
                central_longitude:float=0.0,
                units:str = '',
                return_map:bool = False,
                panel_label:str = None,
                use_robinson = False,
                ):
    '''Plots a global map from swath data.  Input'''
    #construct a 1440,720 map from

    ilats = np.floor((lats + 90.0)/0.25).astype(int)
    ilons = np.floor(lons/0.25).astype(int)

    tot_map = np.zeros((720,1440))
    num_map = np.zeros((720,1440))

    num_map[ilats,ilons] += 1.0
    tot_map[ilats,ilons] += a

    map = tot_map/num_map

    return plot_global_map(map, 
                fig_in = fig_in,
                ax_in = ax_in,
                extent = extent,
                panel_label_loc = panel_label_loc,
                vmin=vmin,
                vmax=vmax, 
                cmap=cmap,
                discrete_cmap=discrete_cmap,
                norm=norm,
                num_levels=num_levels,
                boundaries=boundaries,
                plt_colorbar=plt_colorbar,
                plt_coastlines=plt_coastlines,
                title=title,
                central_longitude=central_longitude,
                units=units,
                return_map=return_map,
                panel_label=panel_label,
                use_robinson=use_robinson,
                )

def global_map(a, 
                fig_in = None,
                ax_in = None,
                extent = None,
                panel_label_loc = [0.03,0.90],
                vmin:float=0.0, 
                vmax:float=30.0, 
                cmap:str = 'BrBG', 
                norm=None,
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
                norm=None,
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
                discrete_cmap = False,
                norm=None,
                num_levels = 10,
                boundaries = None,
                plt_colorbar:bool = False,
                color_bar_orientation = 'horizontal',
                plt_coastlines:bool = True,
                title:str='',
                central_longitude:float=0.0,
                units:str = '',
                return_map:bool = False,
                panel_label:str = None,
                use_robinson = False,
                ):


    img_extent = [-180.0, 180.0, -90.0, 90.0]
    if fig_in is None:
        fig = plt.figure(figsize=(10, 5))  # type: Figure
    else:
        fig = fig_in

    if ax_in is None:
        if use_robinson:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude = central_longitude),title=title)
        else:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = central_longitude),title=title)
        
    else:
        ax = ax_in

    for item in ([ax.title]):
        item.set_fontsize(16)
    sz = a.shape
    num_lons = sz[1]
    if norm is None:
        if discrete_cmap:
            cmap_copy = plt.get_cmap(cmap).copy()
            cmap_list = [cmap_copy(i) for i in range(cmap_copy.N)]

            cmap2 = mpl.colors.LinearSegmentedColormap.from_list('cmap_segmented', cmap_list,cmap_copy.N)
            if boundaries is None:
                boundaries = np.linspace(vmin,vmax,num_levels+1)
            
            norm = mpl.colors.BoundaryNorm(boundaries,cmap2.N,extend='both')
        else:
            cmap_copy = plt.get_cmap(cmap).copy()
            cmap_copy.set_bad('grey')
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap_copy = plt.get_cmap(cmap).copy()
        cmap_copy.set_bad('grey')


    map = ax.imshow(np.flipud(np.roll(a, int(num_lons/2), axis=1)), cmap=cmap_copy, origin='upper', transform=ccrs.PlateCarree(),
                    norm=norm, extent=img_extent)
    if plt_colorbar:
        if discrete_cmap:
            cbar = fig.colorbar(map, shrink=0.7, orientation=color_bar_orientation,extend='both')
            cbar.ax.tick_params(labelsize=14)
        else:
            cbar = fig.colorbar(map, shrink=0.7, orientation=color_bar_orientation,extend='both')
            cbar.ax.tick_params(labelsize=14)
        
    if plt_coastlines:
        try:
            ax.coastlines()
        except:
            print('Trouble getting coastline file')

    ax.set_title(title)
    ax.set_global()

    ax.text(0.5, -0.2, units, va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes)
    if extent is not None:
        ax.set_extent(extent,crs = ccrs.PlateCarree())

    if panel_label is not None:
        plt.text(panel_label_loc[0],panel_label_loc[1],panel_label,transform=ax.transAxes,fontsize=11,
                 bbox={"facecolor": 'grey',"edgecolor": 'grey'})

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

def plot_multiple_maps_with_common_colorbar(maps,
                                            nrows=1,
                                            ncols=1,
                                            vmin=0.0,
                                            vmax=1.0,
                                            titles = None,
                                            panel_labels = None,
                                            panel_label_loc=[0.03,0.87],
                                            cbar_label='',
                                            cmap='BrBG',
                                            ):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from rss_plotting.global_map import plot_global_map

    num_maps = len(maps)
    assert(num_maps == ncols*nrows)

    if titles is None:
        titles = ['' for i in range(num_maps)]
    if panel_labels is None:
        panel_labels = ['' for i in range(num_maps)]

    xsize = 2+ncols*3
    ysize = 2+nrows*1.8
    fig = plt.figure(figsize=(xsize,ysize))
    axs = []
    for label in [titles,panel_labels]:
        if label is None:
            label = ['']*num_maps
    
    for icol in range(ncols):
        for irow in range(0,nrows):
            index = 1 + icol + ncols*irow
            #print(icol,irow,index)
            axs.append(fig.add_subplot(nrows, ncols, index, projection=ccrs.PlateCarree(central_longitude = 0.0))) 
    #print
    for imap,map in enumerate(maps):
        #print(imap,num_maps)
        if imap < (num_maps-1):
            fig,axs[imap] = plot_global_map(map,
                                            vmin=vmin,vmax=vmax,
                                            title=titles[imap],
                                            panel_label=panel_labels[imap],
                                            fig_in=fig,
                                            ax_in=axs[imap],
                                            plt_colorbar=False,
                                            cmap=cmap,
                                            panel_label_loc=panel_label_loc,
                                            return_map=False)
        else:
            fig,axs[imap],im = plot_global_map(map,
                                            vmin=vmin,vmax=vmax,
                                            title=titles[imap],
                                            panel_label=panel_labels[imap],
                                            fig_in=fig,
                                            ax_in=axs[imap],
                                            plt_colorbar=False,
                                            cmap=cmap,
                                            panel_label_loc=panel_label_loc,
                                            return_map=True)

    fig.subplots_adjust(bottom=0.20,top=0.95,left = 0.1,right=0.9,hspace=0.03,wspace=0.05)
    cb_ax = fig.add_axes([0.2,0.10,0.6,0.03])
    cbar = fig.colorbar(im,cax=cb_ax,orientation='horizontal')
    cbar.set_label(cbar_label,fontsize=16)
    cb_ax.tick_params(axis='x', labelsize=14)

    return fig,axs

def plot_multiple_maps_with_different_colorbar(maps,
                                            nrows=1,
                                            ncols=1,
                                            vmin=0.0,
                                            vmax=1.0,
                                            titles = None,
                                            panel_labels = None,
                                            panel_label_loc=[0.03,0.85],
                                            cbar_labels='',
                                            cmap='BrBG',
                                            ):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from rss_plotting.global_map import plot_global_map

    num_maps = len(maps)
    assert(num_maps == ncols*nrows)

    if titles is None:
        titles = ['' for i in range(num_maps)]
    if panel_labels is None:
        panel_labels = ['' for i in range(num_maps)]

    xsize = 2+ncols*3
    ysize = 2+nrows*2.0
    fig = plt.figure(figsize=(xsize,ysize))
    axs = []
    for label in [titles,panel_labels]:
        if label is None:
            label = ['']*num_maps
    
    for icol in range(ncols):
        for irow in range(0,nrows):
            index = 1 + icol + ncols*irow
            #print(icol,irow,index)
            axs.append(fig.add_subplot(nrows, ncols, index, projection=ccrs.PlateCarree(central_longitude = 0.0))) 
    #print
    map_out_list = []
    for imap,map in enumerate(maps):
        #print(imap,num_maps)

        fig,axs[imap],map_out = plot_global_map(map,
                                            vmin=vmin[imap],vmax=vmax[imap],
                                            title=titles[imap],
                                            panel_label=panel_labels[imap],
                                            fig_in=fig,
                                            ax_in=axs[imap],
                                            plt_colorbar=False,
                                            cmap=cmap,
                                            panel_label_loc=panel_label_loc,
                                            return_map=True)
        map_out_list.append(map_out)

    fig.subplots_adjust(bottom=0.10,top=0.95,left = 0.1,right=0.9,hspace=0.6,wspace=0.15)
    
    for k,out_map in enumerate(map_out_list):
        if k < 20:
            locs = axs[k].get_position().bounds

            yloc = locs[1] - 0.16*locs[3]
            xloc = locs[0] + 0.1*locs[2]
            print(xloc,yloc)
            cb_ax = fig.add_axes([xloc,yloc,0.28,0.008])
            cbar = fig.colorbar(out_map,cax=cb_ax,orientation='horizontal')
            cbar.set_label(cbar_labels[k],fontsize=8)
            cb_ax.tick_params(axis='x', labelsize=8)
    return fig,axs