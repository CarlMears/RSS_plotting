def plot_2d_array(a, xvals, yvals,zrange=(0.0,1.0), title='', xtitle='', ytitle='',cmap='BrBG',plt_colorbar = True,zlabel=' '):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    import copy



    X, Y = np.meshgrid(xvals, yvals)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title=title, xlabel=xtitle, ylabel=ytitle)
    im = ax.pcolormesh(X, Y, a, cmap=cmap, norm=colors.Normalize(vmin=zrange[0], vmax=zrange[1]))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    if plt_colorbar:
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(zlabel, fontsize=16)
    return fig,ax