# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:23:19 2018

@author: mears
"""

import matplotlib.colors as colors
import numpy as np

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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

    
def compare_sea_ice_concentration_maps(sea_ice1,sea_ice2,land_mask,label1 = 'Bootstrap',label2 = 'RSS',
                                       ice_thres=0.8,sea_thres=0.2,agree_thres = 0.2,date_str = '',
                                       figpath = 'B:/job_WSF-M/sea_ice/plots/',save_figs = False,save_as_PS = False,
                                       plot_histogram = True,plot_agreement_map =False,
                                       plot_binned_stats=False, plot_diff_hist=False,
                                       use_mean_for_binning = False,
                                       figsize_in=(8,6),requirement=0.0,plot_num_in_bins=False):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.colors as colors
    from binned_stats import calc_binned_means,plot_binned_means

    sea_ice1[sea_ice1 < -10.0] = np.nan
    sea_ice1[sea_ice1 > 10.0] = np.nan
    sea_ice2[sea_ice2 < -10.0] = np.nan
    sea_ice2[sea_ice2 > 10.0] = np.nan

    sz1 = sea_ice1.shape
    sz2 = sea_ice2.shape
    
    if np.any(sz1 != sz2):
        print('maps are not the same size')
        return

    pole = 1
    pole_str = 'NP'
    if sz1[0] == 332:
        pole = 2
        pole_str = 'SP'

    
    
    agree_map = np.zeros((sz1[0],sz1[1]))
    zero_error_map = np.zeros((sz1[0],sz1[1]))
    z = (land_mask == 0.0)
    
    
    hist,xedges,yedges = np.histogram2d(sea_ice1[z], sea_ice2[z], bins=40, range=[[0.0,1.0],[0.0,1.0]])
    
    if plot_histogram:
        fig = plot_2d_sea_ice_hist(hist,xedges,yedges,ice_thres=ice_thres,sea_thres=sea_thres,agree_thres = agree_thres, figsize_in = figsize_in,label1 = label1,label2 = label2)
        if save_figs:
            fig.savefig(figpath +'concentration_2d_hist_'+pole_str+'_'+label1+'_'+label2+'.'+date_str+'.png')
            if save_as_PS:
                fig.savefig(figpath +'concentration_2d_hist_'+pole_str+'_'+label1+'_'+label2+'.'+date_str+'.ps')
        plt.close(fig)

    with np.errstate(invalid='ignore'):
        # region5 is sea_ice 1 > sea ice 2
        z_region5 = np.all([(sea_ice1 >= 0.0),(sea_ice2 >= 0.0),(sea_ice1 - sea_ice2 > agree_thres)],axis=(0))
        agree_map[z_region5] = 5.0
        num_region5 = z_region5.sum()
        region_label5 = 'Large Error, '+label1+' > '+label2

        z_region4 = np.all([(sea_ice1 >= 0.0),(sea_ice2 >= 0.0),(sea_ice1 - sea_ice2 < -agree_thres)],axis=(0))
        agree_map[z_region4] = 4.0
        num_region4 = z_region4.sum()
        region_label4 = 'Large Error, '+label2+' > '+label1

        z_region1 = np.all([(sea_ice1 > ice_thres),(sea_ice2 > ice_thres)],axis=(0))
        agree_map[z_region1] = 1.0
        num_region1 = z_region1.sum()
        if num_region1 < 300:
            print
        region_label1 = 'Ice'

        z_region2 = np.all([(sea_ice1 < sea_thres),(sea_ice1 >= 0.0),
                    (sea_ice2 < sea_thres), (sea_ice2 >= 0.0)],axis=(0))
        agree_map[z_region2] = 2.0
        num_region2 = z_region2.sum()
        region_label2 = 'Ocean'

        not1 = np.any([(sea_ice1 >= sea_thres),(sea_ice2 >= sea_thres)],axis=(0))
        not2 = np.any([(sea_ice1 <= ice_thres),(sea_ice2 <= ice_thres)],axis=(0))
        lt_thres = (np.abs(sea_ice1 - sea_ice2) <= agree_thres)

        z_region3 = np.all([lt_thres,not1,not2],axis=(0))
        agree_map[z_region3] = 3.0
        num_region3 = z_region3.sum()
        region_label3 = 'Mixed, with agreement'
        
        z1 = np.all([(sea_ice1 == 0.0),(sea_ice2 >= 0.05)],axis=(0))
        z2 = np.all([(sea_ice2 == 0.0),(sea_ice1 >= 0.05)],axis=(0))
        
        zero_error_map[z1] = 1.0
        zero_error_map[z2] = 0.5

        agree_map[land_mask == 1] = 0.0
        
    #plot_polar_stereographic(agree_map,figsize=(10,8),coast_color='red',zrange=[0,4])
    
    print(num_region1,num_region2,num_region3,num_region4,num_region5)
    
    classification_labels = ['Land/nodata',region_label1,region_label2,region_label3,region_label4,region_label5]

    '''
    if plot_agreement_map:
        fig2 = plot_sea_ice_agreement(agree_map,num_classes=5,title ='Classification Agreement',figsize_in=figsize_in,classification_labels=classification_labels)
        if save_figs:
            fig2.savefig(figpath +'agreement_map_'+label1+'_'+label2+'.'+date_str+'.png')
            if save_as_PS:
                fig2.savefig(figpath + 'agreement_map_' + label1 + '_' + label2 + '.' + date_str + '.ps')
        plt.close(fig2)
    '''
    #plot binned mean differences, stddevs
    if (use_mean_for_binning):
        binned_stats = calc_binned_means(0.5*(sea_ice1+sea_ice2),sea_ice2-sea_ice1,land_mask,bins=40,xrng = [0.0,1.0],return_as_xarray=True)
    else:
        binned_stats = calc_binned_means(sea_ice1,sea_ice2-sea_ice1,land_mask,bins=40,xrng = [0.0,1.0],return_as_xarray=True)

    z = (land_mask == 0.0)
    hist_diff,xedges_diff,yedges_diff = np.histogram2d(sea_ice1[z],sea_ice2[z]-sea_ice1[z],bins=[40,40], range=[[0.0,1.0],[-0.5,0.5]])    
    if plot_diff_hist:
        fig = plot_2d_sea_ice_diff_hist(hist_diff,xedges_diff,yedges_diff,figsize_in = figsize_in,label1 = label1,label2 = label2)
        plt.close(fig)
    if plot_binned_stats:
        fig20 = plot_binned_means(binned_stats,xlab = label1,ylab =label2+' - '+label1,requirement = requirement,plot_num_in_bins=plot_num_in_bins)
        if save_figs:
            png_file = figpath +'binned_means_'+pole_str+'_'+label1+'_'+label2+'.'+date_str+'.png'
            fig20.savefig(png_file)
            if save_as_PS:
                png_file = figpath + 'binned_means_' + label1 + '_' + label2 + '.' + date_str + '.ps'
                fig20.savefig(png_file)
            plt.close(fig20)

    #print(binned_stats['overall_bias'],binned_stats['overall_std'],binned_stats['overall_rms'])
    return_dict = {'hist':hist,
                   'binned_stats':binned_stats,
                   'agree_map':agree_map,
                   'xedges':xedges,
                   'yedges':yedges,
                   'hist_diff':hist_diff,
                   'xedges_diff':xedges_diff,
                   'yedges_diff':yedges_diff
                   }
           
    return return_dict
    
    

    
def plot_sea_ice_agreement(agree_map,num_classes=5,title ='',classification_labels = ['Land/nodata','Ice','Ocean','Mixed w/agreement','Large Error','Large Error'],figsize_in = (8.,6.)):
 #   from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as colors

    levels = np.linspace(-0.5,5.5,num=7)
    cmap = colors.ListedColormap(['darkblue','white','blue','aqua','orange','red'])
    norm = BoundaryNorm(levels, ncolors=6)

    fig=plt.figure(figsize=figsize_in)
    
    ax = plt.axes(projection=ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0))
    ax.coastlines(resolution='110m',linewidth=0.5,color='w')
    ax.set_extent([5800000,-5800000,-5800000,5800000],crs=ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0))
    ax.gridlines()

    dx = dy = 25000
    x = np.arange(-3850000, +3750000, +dx)
    y = np.arange(+5850000, -5350000, -dy)
    
    #this is a kludge to fill in the background with the open ocean color
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468,468)) 
    background = background + 0.2
    im = ax.pcolormesh(x2, y2, background, cmap=cmap,norm=norm)
    im = ax.pcolormesh(x, y, agree_map, cmap=cmap,norm=norm)
    cbar = fig.colorbar(im, ticks=range(0,len(classification_labels)))
    cbar.ax.set_yticklabels(classification_labels)  # horizontal colorbar
    plt.title(title,fontsize = 20)
    return fig
    
def plot_emiss_histograms_old(edges,y,y2,xtitle,ytitle,title,ylabel1,ylabel2):
    
    import matplotlib.pyplot as plt
    import numpy as np
    fig=plt.figure(figsize=(10,8))
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    Y = np.array([y,y]).T.flatten()
    Y2 = np.array([y2,y2]).T.flatten()
    ax = fig.add_subplot(111, title=title,xlabel = xtitle,ylabel=ytitle)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    ax.plot(X,Y,label=ylabel1)
    ax.plot(X,Y2,label=ylabel2)
    ax.legend()
    return fig
    
def plot_emiss_histogram_array(edges,h,xtitle,ytitle,title,ylabels):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_hists = h.shape[1]
    
    fig=plt.figure(figsize=(10,8))
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    ax = fig.add_subplot(111, title=title,xlabel = xtitle,ylabel=ytitle)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    for ihist in range(0,num_hists):
        y = h[:,ihist]
        Y = np.array([y,y]).T.flatten()
        ax.plot(X,Y,label=ylabels[ihist])
    ax.legend(loc='upper left',fontsize=16)
    return fig
    
def plot_emiss_histogram_array_small(edges,h,xtitle,ytitle,title,ylabels):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_hists = h.shape[1]
    
    fig=plt.figure(figsize=(5,4))
    left,right = edges[:-1],edges[1:]
    X = np.array([left,right]).T.flatten()
    ax = fig.add_subplot(111, title=title,xlabel = xtitle,ylabel=ytitle)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(14)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    for ihist in range(0,num_hists):
        y = h[:,ihist]
        Y = np.array([y,y]).T.flatten()
        ax.plot(X,Y,label=ylabels[ihist])
    ax.legend(loc='upper left',fontsize=12)
    return fig

    
def plot_sea_ice_concentration(seainc_conc,title ='',proj = 'Stereographic',fsize =(8,6),units = 'Sea Ice Concentration'):
 #   from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
 #   import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean

    levels = MaxNLocator(nbins=100).tick_values(0.0,1.0)
    cmap = cmocean.cm.ice
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    fig=plt.figure(figsize=fsize)
    
    #ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0))
    proj_not_found = True
    if proj == 'Stereographic':
        projection = ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0)
        proj_not_found = False
    if proj == 'EASE2':
        projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-45.0, central_latitude=90.0)
        proj_not_found = False
         
    if proj_not_found:
        print('No match to Projection ',proj)
        return
    
    ax = plt.axes(projection=projection)
    ax.coastlines(resolution='50m',linewidth=0.5,color='red')
    ax.set_extent([5800000,-5800000,-5800000,5800000],crs=projection)
    ax.gridlines()

    dx = dy = 25000
    x = np.arange(-3850000, +3750000, +dx)
    y = np.arange(+5850000, -5350000, -dy)
    
    #this is a kludge to fill in the background with the open ocean color
    x2 = np.arange(-5850000, +5850000, +dx)
    y2 = np.arange(+5850000, -5850000, -dy)
    background = np.zeros((468,468)) 
    background = background + 0.2
    
    im = ax.pcolormesh(x2, y2, background, cmap=cmap, norm=norm)
    im = ax.pcolormesh(x, y, seainc_conc, cmap=cmap, norm=norm)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(14)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
 
    cbar = fig.colorbar(im)
    cbar.set_ticks(np.linspace(0.0,1.0,11))
    cbar.ax.set_ylabel(units,fontsize=14)
    for item in (cbar.ax.get_yticklabels()):
        item.set_fontsize(12)

    plt.title(title,fontsize=20)
    return fig
    
def calc_2D_sea_ice_emiss_hist(seaice,emiss,land_mask2):

    import numpy as np
    
    z = (land_mask2 == 0.0)
    hist,xedges,yedges = np.histogram2d(seaice[z], emiss[z], bins=120, range=[[0.0,1.1999],[0.0,1.1999]])
    hist = hist.T
    hist[0,0] = 0.0
    return hist,xedges,yedges
    
def plot_2D_sea_ice_emiss_hist(hist,xedges,yedges,title='',xtitle='Sea Ice Concetration',ytitle='Emissivity',figsize_in=(7,5),vmax = -999.0,vmin = -999.0):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
        
    X, Y = np.meshgrid(xedges, yedges)
    fig=plt.figure(figsize=figsize_in)
    #ax = fig.add_subplot(111, title=title,aspect='equal',xlabel = xtitle,ylabel=ytitle)
    ax = fig.add_subplot(111, title=title,xlabel = xtitle,ylabel=ytitle)
    exp10 = np.floor(np.log10(hist.max()))

    if vmax == -999.0:
        vmax = 10**exp10

    if vmin == -999.0:
        vmin = np.max([vmax/10000.0,1.0])

    im = ax.pcolormesh(X, Y, hist,cmap = 'ocean_r',norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    #ax.set_xlim([0.0,1.1])
    #ax.set_ylim([0.0,1.1])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    
#    emiss_median = np.zeros(10)
#    delta_x = 0.1
#    x_median = np.arange(delta_x/2.0,1.0,delta_x)
#    lower_x = np.arange(0.0,1.0,delta_x)
#    for i,x_lower in enumerate(lower_x):
#        z = np.all([(land_mask2 == 0),(seaice >= x_lower),(seaice < (x_lower+delta_x)),(emiss > 0.01)],axis = (0))
#        emiss_median[i] = np.median(emiss[z])
#    
#    emiss_median[emiss_median > 1.0] = 1.0
#    ax.plot(x_median,emiss_median,marker='o',color='r')
    
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Number of Observations',fontsize=14)
    return fig
    


def plot_2D_arr(arr, xedges, yedges, title='', xtitle='', ytitle='',units='', nx=366,ny=40, zrange=(0.0, 1.2),
                     z2_range=(0.0, 1.2),cmap = 'BrBG'):
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import numpy as np

        X, Y = np.meshgrid(xedges, yedges)
        arr_max = np.max(arr)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, title=title, xlabel=xtitle, ylabel=ytitle)
        im = ax.pcolormesh(X, Y, arr, norm=colors.Normalize(vmin=zrange[0],vmax=zrange[1]),cmap=cmap)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(20)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)

        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(units, fontsize=16)

        return fig
    
#    emiss_median = np.zeros(10)
#    delta_x = 0.1
#    x_median = np.arange(delta_x/2.0,1.0,delta_x)
#    lower_x = np.arange(0.0,1.0,delta_x)
#    for i,x_lower in enumerate(lower_x):
#        z = np.all([(land_mask2 == 0),(seaice >= x_lower),(seaice < (x_lower+delta_x)),(emiss > 0.01)],axis = (0))
#        emiss_median[i] = np.median(emiss[z])
#    
#    emiss_median[emiss_median > 1.0] = 1.0
#    ax.plot(x_median,emiss_median,marker='o',color='r')

	

    
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
    z = np.flipud(z)
    print('Caution - new version of PPS with flip up down')

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
    
def compare_sea_ice_concentration_cumulative(sea_ice1,sea_ice2,land_mask,hist_in = False,agree_num_map_in=False,
                                             binned_stats_in=False,ice_thres=0.8,sea_thres=0.2,agree_thres = 0.2):
    

    import numpy as np
    import matplotlib.pyplot as plt
    
    sz1 = sea_ice1.shape
    sz2 = sea_ice2.shape
    
    if np.any(sz1 != sz2):
        print('maps are not the same size')
        return
    
    
    agree_num_map = np.zeros((sz1[0],sz1[1],5))
    
    z = (land_mask == 0.0)
    hist,xedges,yedges = np.histogram2d(sea_ice1[z], sea_ice2[z], bins=40, range=[[0.0,1.0],[0.0,1.0]])
    
    try:
        hist = hist + hist_in
    except:
        print
    

    z_region5 = np.all([(sea_ice1 >= 0.0),(sea_ice2 >= 0.0),(sea_ice1 - sea_ice2 > agree_thres)],axis=(0))
    agree_num_map[z_region5,4] = agree_num_map[z_region5,4]+1

    z_region4 = np.all([(sea_ice1 >= 0.0),(sea_ice2 >= 0.0),(sea_ice1 - sea_ice2 < -agree_thres)],axis=(0))
    agree_num_map[z_region4,3] = agree_num_map[z_region4,3]+1
 

    z_region1 = np.all([(sea_ice1 > ice_thres),(sea_ice2 > ice_thres)],axis=(0))
    agree_num_map[z_region1,0] = agree_num_map[z_region1,0] + 1

    z_region2 = np.all([(sea_ice1 < sea_thres),(sea_ice1 >= 0.0),(sea_ice2 < sea_thres), (sea_ice2 >= 0.0)],axis=(0))
    agree_num_map[z_region2,1] = agree_num_map[z_region2,1] + 1


    not1 = np.any([(sea_ice1 >= sea_thres),(sea_ice2 >= sea_thres)],axis=(0))
    not2 = np.any([(sea_ice1 <= ice_thres),(sea_ice2 <= ice_thres)],axis=(0))
    lt_thres = (np.abs(sea_ice1 - sea_ice2) <= 0.3)

    z_region3 = np.all([lt_thres,not1,not2],axis=(0))
    agree_num_map[z_region3,2] = agree_num_map[z_region3,2] + 1
    
    try:
        agree_num_map = agree_num_map + agree_num_map_in
    except:
        print
    
    #plot binned mean differences, stddevs
    
    binned_stats = calc_binned_means(sea_ice1,sea_ice2-sea_ice1,land_mask,bins=40,xrng = [0.0,1.0])
    
    try:
        binned_stats = combine_binned_means(binned_stats,binned_stats_in)
    except:
        raise ValueError('need to initialize binned stats')
    
    return hist,xedges,yedges,binned_stats,agree_num_map
        
def plot_2d_sea_ice_hist(hist_in,xedges,yedges,ice_thres=0.8,sea_thres=0.2,agree_thres = 0.2, figsize_in = (8,6),label1 = '',label2 = '',vmin = -999.0,vmax = -999.0,plot_regions=False,plot_agree=True,title='Ice Concentration Comparison'):
        
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.colors as colors


    
    X, Y = np.meshgrid(xedges, yedges)
    hist = hist_in.T

    exp10 = np.floor(np.log10(hist.max()))

    if vmax == -999.0:
        vmax = 10 ** exp10

    if vmin == -999.0:
        vmin = np.max([vmax / 10000.0, 1.0])

    fig=plt.figure(figsize=figsize_in)
    ax = fig.add_subplot(111, title=title,aspect='equal',xlabel = label1,ylabel=label2)
    
    hist_plot = ax.pcolormesh(X, Y, hist,cmap = 'ocean_r',norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    #draw diagonal line
    ax.plot([0.0,1.0],[0.0,1.0])

    # draw lines for substantial error threshold
    if (plot_agree or plot_regions):
        ax.plot([agree_thres, 1.0], [0.0, 1.0 - agree_thres], 'r')
        ax.plot([0.0, 1.0 - agree_thres], [agree_thres, 1.0], 'r')

    if plot_regions:
        ax.add_patch(Polygon([(agree_thres,0.0), (1.0, 0.0), (1.0,1.0-agree_thres), (agree_thres,0.0)],
                              closed=True, facecolor='red',alpha=0.1))
        ax.add_patch(Polygon([(0.0,agree_thres), (1.0-agree_thres,1.0), (0.0,1.0), (0.0,agree_thres)],
                           closed=True, facecolor='red',alpha=0.1))
    
        ax.add_patch(Polygon([(0.0,sea_thres),(sea_thres,sea_thres),(sea_thres,0.0),(agree_thres,0.0), (1.0,1.0-agree_thres),(1.0,ice_thres),(ice_thres,ice_thres),(ice_thres,1.0), (1.0-agree_thres,1.0),(0.0,agree_thres),(0.0,sea_thres)],
                           facecolor='yellow',alpha=0.1))
                           
        ax.add_patch(Polygon([(0.0,0.0),(0.0,sea_thres), (sea_thres,sea_thres), (sea_thres,0.0), (0.0,0.0)],
                           closed=True, facecolor='blue',alpha=0.2))                       
        ax.plot([sea_thres,sea_thres,0.0],[0.0,sea_thres,sea_thres],'b')
        ax.plot([ice_thres,ice_thres,1.0],[1.0,ice_thres,ice_thres],'black')

        ax.text(0.1,0.8,'Region 4',fontsize=14)
        ax.text(0.8,0.2,'Region 5',fontsize=14)
        ax.text(0.4,0.6,'Region 3',fontsize=14)
        ax.text(0.03,0.15,'Region 2',fontsize=14)
        ax.text(0.83,0.95,'Region 1',fontsize=14,rotation = 'vertical')
        
    cb = fig.colorbar(hist_plot, ax=ax, extend='max')
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_ylabel('Number of Observations',fontsize=16)

    return fig
    
def plot_2d_sea_ice_diff_hist(hist_in,xedges,yedges, figsize_in = (8,6),label1 = '',label2 = '',vmax = 200.,title='Ice Concentration Difference'):
        
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.colors as colors
    
    X, Y = np.meshgrid(xedges, yedges)
    hist = hist_in.T

    fig=plt.figure(figsize=figsize_in)
    yr = [yedges[0],yedges[-1]]

    ax = fig.add_subplot(111, title=title,aspect='equal',xlabel = label1,ylabel=label2)
    ax.set_ylim(yedges[0],yedges[-1])
    hist_plot = ax.pcolormesh(X, Y, hist,cmap = 'ocean_r',norm=colors.LogNorm(vmin=0.5, vmax=vmax))
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    #draw horizontal line
    ax.plot([0.0,1.0],[0.0,0.0])
           
    cb = fig.colorbar(hist_plot, ax=ax, extend='max')
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_ylabel('Number of Observations',fontsize=16)

    return fig

def plot_polar_stereographic(z_in,subplot_arg=111,fig = None,title ='',figsize = (13, 10),zrange=(0.0,1.0),cmap = None,units = 'Concentration',coast_color = 'w',land = None,land_color='tan',pole='north'):

    if pole == 'north':
        return plot_polar_stereographic_NP(z_in,
                                subplot_arg=subplot_arg,
                                fig = fig,
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
                                subplot_arg=subplot_arg,
                                fig = fig,
                                title =title,
                                figsize = figsize,
                                zrange=zrange,
                                cmap = cmap,
                                units = units,
                                coast_color = coast_color,
                                land = land,
                                land_color=land_color)
    
def plot_polar_stereographic_NP(z_in,subplot_arg=111,fig = None,title ='',figsize = (13, 10),zrange=(0.0,1.0),cmap = None,units = 'Concentration',coast_color = 'w',land = None,land_color='tan'):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import cmocean
    import copy

    #no side effects
    z = copy.copy(z_in)
    
    #z = np.flipud(z)
    #print('Caution - new version of PPS with flip up down')
    
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
       
    if fig is None: 
        fig=plt.figure(figsize=figsize)
    
    projection = ccrs.Stereographic(central_latitude=90.0, central_longitude=-45.0,true_scale_latitude=70.0)
    ax = fig.add_subplot(subplot_arg,projection=projection)
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
    return fig


def plot_polar_stereographic_SP(z_in, subplot_arg=111, fig=None, title='', figsize=(13, 10), zrange=(0.0, 1.0), cmap=None,
                             units=None, coast_color='w', land=None,land_color='tan'):
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

    if fig is None:
        fig = plt.figure(figsize=figsize)

    projection = ccrs.Stereographic(central_latitude=-90.0, central_longitude=0.0, true_scale_latitude=70.0)
    ax = fig.add_subplot(subplot_arg, projection=projection)
    ax.coastlines(resolution='50m', linewidth=0.5, color=coast_color)
    ax.set_extent([5800000, -5800000, -5800000, 5800000], crs=projection)
    ax.gridlines()

    l, b, w, h = ax.get_position().bounds

    dx = dy = 25000
    x = np.arange(-3950000 + dx/2, 3950000,  dx)
    y = np.arange(-3950000 + dy/2,+4350000,  dy)
    #print(np.min(x),np.max(x))
    #print(np.min(y),np.max(y))
    # this is a kludge to fill in the background with the open ocean color
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
 