
def plt_zonal_mean_array(zonal_mean_array,
                         time_range = [2007,2018],
                         title = 'Zonal Means',
                         vmin = -2.0,
                         vmax =   2.0,
                         cmap='BrBG',
                         plt_colorbar=True):

    # plot a color-coded map of the zonal mean array (hofmeuller plot)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
 
    img_extent = [time_range[0],time_range[1], -90.0, 90.0]
    fig = plt.figure(figsize=(10, 5))  # type: Figure
    ax = fig.add_subplot(1, 1, 1,title=title)
    for item in ([ax.title]):
        item.set_fontsize(16)
    lats = -90.0 + 0.125 + np.arange(720)/4.0
    yrs = time_range[0] + 1.0/24.0 + (1.0/12.0)*np.arange(0,12*(time_range[1]-time_range[0]))
    xx, yy = np.meshgrid(yrs, lats)
    c = ax.pcolormesh(xx, yy, zonal_mean_array.T, cmap='RdBu', vmin=vmin, vmax=vmax)
    if plt_colorbar:
        cbar = fig.colorbar(c, shrink=0.7, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
    return fig, ax


def plt_zonal_mean_array_fixed_axis(zonal_mean_array,
                         time_range=[2007, 2018],
                         title='Zonal Means',
                         vmin=-2.0,
                         vmax=2.0,
                         cmap='BrBG',
                         plt_colorbar=True):
    # plot a color-coded map of the zonal mean array (hofmeuller plot)
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import Divider, Size
    from mpl_toolkits.axes_grid1.mpl_axes import Axes
    import matplotlib as mpl
    fig = plt.figure(figsize=(17, 3))
    h_start = 0.5*(time_range[0] - 1988.0)
    h_end = 0.5*(time_range[1]-1988.0)
    print(h_start,h_end,h_end-h_start)

    h = [Size.Fixed(1.0+h_start),Size.Fixed(h_end-h_start)]
    v = [Size.Fixed(0.7),Size.Fixed(2.0)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    fig.add_axes(ax)
    ax.set_title(title)
    ax.set_xlim(time_range)
    ax.set_ylim(-90.0,90.0)
    img_extent = [time_range[0], time_range[1], -90.0, 90.0]
    for item in ([ax.title]):
        item.set_fontsize(16)
    lats = -90.0 + 0.125 + np.arange(720) / 4.0
    yrs = time_range[0] + 1.0 / 24.0 + (1.0 / 12.0) * np.arange(0, 12 * (time_range[1] - time_range[0]))
    xx, yy = np.meshgrid(yrs, lats)
    c = ax.pcolormesh(xx, yy, zonal_mean_array.T, cmap='RdBu', vmin=vmin, vmax=vmax)
    #if plt_colorbar:
     #   cbar = fig.colorbar(c, shrink=0.7, orientation='horizontal')
    #    cbar.ax.tick_params(labelsize=14)
    return fig, ax