def global_map(a, vmin=0.0, vmax=30.0, cmap=None, plt_colorbar=False,title='',extent=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.colors as colors

    img_extent = [-180.0, 180.0, -90.0, 90.0]
    fig = plt.figure(figsize=(10, 5))  # type: Figure
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(),title=title)
    for item in ([ax.title]):
        item.set_fontsize(16)
    map = ax.imshow(np.flipud(np.roll(a, 720, axis=1)), cmap=cmap, origin='upper', transform=ccrs.PlateCarree(),
                    norm=colors.Normalize(vmin=vmin, vmax=vmax), extent=img_extent)
    if plt_colorbar:
        cbar = fig.colorbar(map, shrink=0.7, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
    ax.coastlines()
    ax.set_global()
    if extent is not None:
        ax.set_extent(extent)
    return fig, ax

def plot_scat(x,y, title='', xtitle='', ytitle='', nbins=120, x_range=None,
                 y_range=None, aspect='equal', plot_diagonal=True):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title=title, aspect=aspect, xlabel=xtitle, ylabel=ytitle)

    ax.scatter(x,y,marker='+')
    if ~(x_range is None):
        ax.set_xlim(x_range)
    if ~(y_range is None):
        ax.set_ylim(y_range)
    return fig,ax

def plot_binned_means_by_ws_and_rain(binned_stats_by_ws_and_rain,
                                     xlabel = 'ERA5 Wind Speed (m/s)',
                                     ylabel = 'Rain Bin (mm/hr)',
                                     rr_labels = None,
                                     plot_std = False):

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    binned_means_2d = binned_stats_by_ws_and_rain.values[1,:,:]/binned_stats_by_ws_and_rain.values[0,:,:]
    binned_std_2d   = (binned_stats_by_ws_and_rain.values[2,:,:] -
                       np.square(binned_stats_by_ws_and_rain.values[1,:,:])/
                       binned_stats_by_ws_and_rain.values[0,:,:])/(binned_stats_by_ws_and_rain.values[0,:,:]-1)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    xedges = binned_stats_by_ws_and_rain.coords['rain_bin']
    yedges = binned_stats_by_ws_and_rain.coords['wind_bin']

    if plot_std:
        #levels = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0]
        levels = MaxNLocator(nbins=30).tick_values(0.0,6.0)
        cmap = plt.get_cmap('viridis')
    else:
        #levels = np.array([-8.0,-5.0,-3.0,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0,5.0,8.0])
        levels = np.array([-8.0,-6.5,-5.0,-4.0,-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,
                           0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,6.5,8.0])
        cmap = plt.get_cmap('BrBG_r')

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if plot_std:
        im = ax.pcolormesh(yedges,xedges,binned_std_2d, cmap=cmap, norm=norm)
    else:
        im = ax.pcolormesh(yedges,xedges,binned_means_2d, cmap=cmap, norm=norm)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params(direction='in')
    ax.set_yticks(xedges[:-1]+0.5*(xedges[1] - xedges[0]))
    if rr_labels is not None:
        ax.set_yticklabels(rr_labels)


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    if plot_std:
        cbar.ax.set_ylabel('Std. Difference (sat - ERA5)', fontsize=16)
        cbar.set_ticks([0.0,2.0,4.0,6.0])
        cbar.ax.tick_params(direction='in')
    else:
        cbar.ax.set_ylabel('Mean Difference (sat - ERA5)', fontsize=16)
        cbar.set_ticks([-8.0,-4.0,-2.0,-1.0,0.0,1.0, 2.0, 4.0, 8.0])
        cbar.ax.tick_params(direction='in')
    return fig


