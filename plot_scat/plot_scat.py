def plot_scat(x,y, title='', xtitle='', ytitle='', nbins=120, x_range=None,
                 y_range=None, aspect='equal', plot_diagonal=True, marker = '+'):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, title=title, aspect=aspect, xlabel=xtitle, ylabel=ytitle)

    ax.scatter(x,y,marker=marker)
    if ~(x_range is None):
        ax.set_xlim(x_range)
    if ~(y_range is None):
        ax.set_ylim(y_range)

    if plot_diagonal:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if (xmax-xmin) >= (ymax-ymin):
            ax.plot([xmin,xmax],[xmin,xmax])
        else:
            ax.plot([ymin,ymax],[ymin,ymax])


    return fig,ax