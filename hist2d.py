import numpy as np
import xarray as xr
import matplotlib.colors as colors
import cmocean
import copy

def hists_compatible(hist2d_1,hist2d_2):

    ''' this functions checks to make sure that two hist2d's have the sam histtogram  sizes and edges
        this should be called beofrore attmpeting to combined them'''

    try:
        shape1 = hist2d_1.histogram.shape
        shape2 = hist2d_2.histogram.shape
        if (shape1 != shape2):
            return False
    except:
        return False

    try:
        if (np.any(hist2d_1.xedges != hist2d_2.xedges)):
            return False
        if (np.any(hist2d_1.yedges != hist2d_2.yedges)):
            return False
    except:
        return False

    return True

def combine_2d_hist_xr(hist2d_1,hist2d_2):

    if hists_compatible(hist2d_1, hist2d_2):
        hist2d = copy.copy(hist2d_1)
        hist2d.histogram.data = hist2d.histogram.data + hist2d_2.histogram.data
        try:
            hist2d.histogram_no_degr.data = hist2d.histogram_no_degr.data + hist2d_2.histogram_no_degr.data
        except:
            print('No degradation_free data')
        return hist2d
    else:
        return False


def plot_2D_hist_xr(hist2d,
                    title='', xtitle='', ytitle='',
                    z1_range=(0.0, 1.0), z2_range=(0.0, 1.0), aspect='equal',
                    vmax=None, vmin=None, plot_diagonal=False,lognorm=False,plot_line = None,
                    cmap = None,figsize=(8, 6),agree_thres=None,var='histogram'):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    
    if vmax is None:
        vmax = np.max(hist2d.histogram)
    if vmin is None:
        vmin = vmax / 1000.0

    try:
        hist = hist2d[var].data
        X, Y = np.meshgrid(hist2d.xedges.data, hist2d.yedges.data)
    except:
        raise(ValueError('hist2d not expected xarray, or var not present'))

    if vmax is None:
        vmax = np.max(hist)
    if vmin is None:
        vmin = vmax / 1000.0
        
    if cmap is None:
        cmap = 'ocean_r'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, title=title, aspect=aspect, xlabel=xtitle, ylabel=ytitle)
    if lognorm:
        im = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.pcolormesh(X, Y, hist, cmap=cmap,vmax=vmax)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    if plot_diagonal:
        ax.plot(z1_range, z2_range, color='r')
    if plot_line is not None:
        x_line = plot_line[1] + plot_line[0]*hist2d.ybin.data
        ax.plot(x_line,hist2d.ybin.data,color='b')

    if agree_thres is not None:
        ax.plot([agree_thres, 1.0], [0.0, 1.0 - agree_thres], 'r')
        ax.plot([0.0, 1.0 - agree_thres], [agree_thres, 1.0], 'r')

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('Number of Observations', fontsize=16)
    return fig


def plot_2D_hist(hist, xedges, yedges, title='', xtitle='', ytitle='', nbins=120, z1_range=(0.0, 1.2),
                 z2_range=(0.0, 1.2), aspect='equal', vmax=None, vmin=None, plot_diagonal=False):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    if vmax is None:
        vmax = np.max(hist)
    if vmin is None:
        vmin = vmax / 1000.0

    X, Y = np.meshgrid(xedges, yedges)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title=title, aspect=aspect, xlabel=xtitle, ylabel=ytitle)
    im = ax.pcolormesh(X, Y, hist, cmap='ocean_r', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    if plot_diagonal:
        ax.plot(z1_range, z2_range, color='r')

    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('Number of Observations', fontsize=16)
    return fig
