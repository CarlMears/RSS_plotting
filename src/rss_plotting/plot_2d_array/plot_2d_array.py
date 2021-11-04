def plot_2d_array(a, 
                  xvals, 
                  yvals,
                  zrange=None, 
                  title='', 
                  xtitle='', 
                  ytitle='',
                  cmap='BrBG',
                  plt_colorbar = True,
                  norm='linear',
                  log_scale = 1000.0,
                  zlabel=' ',
                  fig_in = None,
                  ax_in = None,
                  font_size=13):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import copy

    if fig_in is None:
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = fig_in

    if ax_in is None:
        ax = fig.add_subplot(111)
    else:
        ax = ax_in

    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    X, Y = np.meshgrid(xvals, yvals)
    
    if norm == 'linear':
        if zrange is None:
            zrange=[np.nanmin(a),np.nanmax(a)]
        im = ax.pcolormesh(X, Y, a, shading='auto',cmap=cmap, norm=colors.Normalize(vmin=zrange[0], vmax=zrange[1]))
    elif norm == 'log':
        if zrange is None:
            zrange=[np.nanmin(a),np.nanmax(a)]
        im = ax.pcolormesh(X, Y, a, shading='auto',cmap=cmap, norm=colors.LogNorm(vmin=zrange[1]/log_scale, vmax=zrange[1]))
    else:
        raise ValueError(f'norm = {norm} not supported')
    #shading = 'auto' chooses shading based on sizes of X, Y and a
    #see https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_grids.html

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(font_size+4)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

    if plt_colorbar:
        cbar = fig.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.set_ylabel(zlabel, fontsize=font_size)
    
    return fig,ax