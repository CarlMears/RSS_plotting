# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:23:19 2018

@author: mears
"""

import matplotlib.colors as colors
import numpy as np

def mask_lakes(z_in,mask_value=-999.0,lake_names = ['all']):
    import numpy as np
    
    shp = z_in.shape
    if ((shp[0] == 448) and (shp[1] == 304)):   
        z = np.copy(z_in)
        for lake_name in lake_names:
            if ((lake_name == 'all') or (lake_name == 'great lakes')):
                z[20:85,0:48] = mask_value # great lakes
            if ((lake_name == 'all') or (lake_name == 'winnipeg')):
                z[90:125,0:30] = mask_value # lake winnipeg
            if ((lake_name == 'all') or (lake_name == 'great slave')):
                z[155:185,20:50] = mask_value # great slave lank
            if ((lake_name == 'all') or (lake_name == 'great bear')):
                z[180:200,40:60] = mask_value #great bear lake
            if ((lake_name == 'all') or (lake_name == 'biakal')):
                z[350:370,215:250] =mask_value #biakal
        return z
    else:
        raise ValueError('map wrong size for lake masking')

    
def plot_polar_stereographic_rgb(z_in,subplot_arg=111,fig = None,title ='',figsize = (13, 10),zrange=(0.0,1.0),coast_color = 'w'):
 #   from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
 #   import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean
    import copy
    
    z = copy.copy(z_in)
    z = np.flipud(z)
    print('Caution - new version of PPS with flip up down')
       
    if fig is None: 
        fig=plt.figure(figsize=figsize)
    
    projection = ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0)
    ax = fig.add_subplot(subplot_arg,projection=projection)  #,position = [0.1,0.1,0.9,0.9])
   
    
    ax.coastlines(resolution='50m',linewidth=0.5,color=coast_color)
    ax.set_extent([5800000,-5800000,-5800000,5800000],crs=projection)
    ax.gridlines()

    dx = dy = 25000
    x = np.arange(-3850000+12500, +3750000+12500, +dx)
    y = np.arange(+5850000, -5350000, -dy)
    
    #this is a kludge to fill in the background with the open ocean color
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468,468)) 
    background = background
    
    cmap = copy.copy(cmocean.cm.ice)
    cmap.set_under('grey')
    levels = MaxNLocator(nbins=100).tick_values(0.0,1.0)   
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
 
    im = ax.pcolormesh(x2, y2, background, cmap=cmap, norm=norm)
     
    z_rgb = z[:, :-1, :]
    colorTuple = z_rgb.reshape((z_rgb.shape[0] * z_rgb.shape[1]), 3)
    colorTuple = np.insert(colorTuple,3,1.0,axis=1)

    # What you put in for the image doesn't matter because of the color mapping
    im = ax.pcolormesh(x,y, z[:,:,0], color=colorTuple)
    
    l, b, w, h = ax.get_position().bounds
    ax.set_position([l,b,0.281818,h])

    plt.title(title,fontsize=20)
    return fig
    
def plot_polar_stereographic_listed_colormap(z_in,title ='',subplot_arg=111,coast_color='r',figsize = (13, 10),zrange=(0.0,1.0),color_list = ['black','darkblue','blue','aqua','white'],classification_labels = ['Land/nodata','Ocean','New Ice','First Year','Multi Year']):
 #   from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as colors

    z = np.copy(z_in)
    
    num_categories = len(color_list)
    levels = np.linspace(zrange[0]-0.5,zrange[1] + 0.5,num=num_categories+1)
    
    cmap = colors.ListedColormap(color_list)
    norm = BoundaryNorm(levels, ncolors=num_categories)

    fig=plt.figure(figsize=figsize)
    
    projection = ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0, true_scale_latitude=70.0)
    ax = fig.add_subplot(subplot_arg, projection=projection)
    ax.coastlines(resolution='50m', linewidth=0.5, color=coast_color)
    ax.set_extent([5800000, -5800000, -5800000, 5800000], crs=projection)
    ax.gridlines()

    l, b, w, h = ax.get_position().bounds

    dx = dy = 25000
    x = np.arange(-3850000 + 12500, +3750000, +dx)
    y = np.arange(-5350000 + 1250, +5850000, +dy)

    # this is a kludge to fill in the background with the open ocean color
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468, 468))
    background = background
    #------------------

    im = ax.pcolormesh(x2, y2, background, cmap=cmap,norm=norm)
    im = ax.pcolormesh(x, y, z, cmap=cmap,norm=norm)
    
    cbar = fig.colorbar(im, ticks=np.arange(zrange[0],zrange[1]+1))
    cbar.ax.set_yticklabels(classification_labels) 
    plt.title(title,fontsize = 20)
    return fig

def plot_polar_stereographic(z_in,subplot_arg=111,fig_in = None,ax_in = None,title ='',figsize = (13, 10),zrange=(0.0,1.0),cmap = None,units = 'Concentration',coast_color = 'w',land = None,land_color='tan',pole='north'):

    if pole == 'north':
        return plot_polar_stereographic_NP(z_in,
                                fig_in = fig_in,
                                ax_in = ax_in,
                                title =title,
                                figsize = figsize,
                                zrange=zrange,
                                cmap = cmap,
                                units = units,
                                coast_color = coast_color,
                                land = land,
                                land_color=land_color)
    if pole == 'south':
        return plot_polar_stereographic_SP(z_in,
                                fig_in = fig_in,
                                ax_in = ax_in,
                                title =title,
                                figsize = figsize,
                                zrange=zrange,
                                cmap = cmap,
                                units = units,
                                coast_color = coast_color,
                                land = land,
                                land_color=land_color)
    
def plot_polar_stereographic_NP(z_in,fig_in = None,ax_in=None,title ='',figsize = (13, 10),zrange=(0.0,1.0),cmap = None,units = 'Concentration',coast_color = 'w',land = None,land_color='tan'):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean
    import copy

    #if ax_in is passed in (as for a multi panel plot), the projection needs to be already defined
    # when then the axis was created )

    #no side effects
    z = copy.copy(z_in)
    
    if cmap == None:
        cmap = copy.copy(cmocean.cm.ice)
        cmap.set_under('grey')
        
    levels = MaxNLocator(nbins=100).tick_values(zrange[0],zrange[1])   
    norm   = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    np.clip(z,zrange[0]+0.0001,zrange[1]-0.0001,out=z)

    not_finite = ~np.isfinite(z)
    z[not_finite] = zrange[0] - 0.01*abs(zrange[1] - zrange[0])

    
    if land is not None:   #we have a land mask
        if z.shape == land.shape:  # it is the right size
            z[land] = zrange[1] + 0.01*abs(zrange[1] - zrange[0])
            cmap.set_over(land_color)
       
    if fig_in is None: 
        fig=plt.figure(figsize=figsize)
    else:
        fig = fig_in
    
    projection = ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0)
    if ax_in is None:
        ax = fig.add_subplot(projection=projection)
    else:
        ax = ax_in

    ax.coastlines(resolution='50m',linewidth=0.5,color=coast_color)
    ax.set_extent([5800000,-5800000,-5800000,5800000],crs=projection)
    ax.gridlines()
    
    l, b, w, h = ax.get_position().bounds
   

    dx = dy = 25000
    x = np.arange(-3850000+12500, +3750000, +dx)
    y = np.arange(-5350000+1250,+5850000,  +dy)
    
    #this is a kludge to fill in the background with the open ocean color
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468,468)) 
    background = background
    
    im = ax.pcolormesh(x2, y2, background, cmap=cmap, norm=norm)
    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
     
    cbar = fig.colorbar(im)
    cbar.set_ticks(np.linspace(zrange[0],zrange[1],11))
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(20)
    cbar.ax.set_ylabel(units,fontsize = 24,weight="bold")
    
    plt.title(title,fontsize=20)
    return fig,ax


def plot_polar_stereographic_SP(z_in, 
                                subplot_arg=111, 
                                fig_in=None, 
                                ax_in = None, 
                                title='', 
                                figsize=(13, 10), 
                                zrange=(0.0, 1.0), 
                                cmap=None,
                                units=None, 
                                coast_color='w', 
                                land=None,
                                land_color='tan'):
    #   from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    #   import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean
    import copy

    z = copy.copy(z_in)

    if cmap == None:
        cmap = copy.copy(cmocean.cm.ice)
        cmap.set_under('grey')

    levels = MaxNLocator(nbins=100).tick_values(zrange[0], zrange[1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    #np.clip(z, zrange[0] + 0.0001, zrange[1] - 0.0001, out=z)

    not_finite = ~np.isfinite(z)
    z[not_finite] = zrange[0] - 0.01 * abs(zrange[1] - zrange[0])

    if land is not None:  # we have a land mask
        if z.shape == land.shape:  # it is the right size
            z[land] = zrange[1] + 0.01 * abs(zrange[1] - zrange[0])
            cmap.set_over(land_color)

    if fig_in is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = fig_in

    projection = ccrs.Stereographic(central_latitude=-90.0, central_longitude=0.0, true_scale_latitude=70.0)
    if ax_in is None:
        ax = fig.add_subplot(projection=projection)
    else:
        ax = ax_in

    ax.coastlines(resolution='50m', linewidth=0.5, color=coast_color)
    ax.set_extent([5800000, -5800000, -5800000, 5800000], crs=projection)
    ax.gridlines()

    l, b, w, h = ax.get_position().bounds

    dx = dy = 25000
    x = np.arange(-3950000 + dx/2, 3950000,  dx)
    y = np.arange(-3950000 + dy/2,+4350000,  dy)
    
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468, 468))
    background = background

    im = ax.pcolormesh(x2, y2, background, cmap=cmap, norm=norm)
    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im)
    cbar.set_ticks(np.linspace(zrange[0], zrange[1], 11))
    cbar.ax.set_ylabel(units, fontsize=16)

    plt.title(title, fontsize=20)
    return fig
 