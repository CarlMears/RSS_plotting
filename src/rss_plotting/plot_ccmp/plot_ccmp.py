import os
import datetime
import argparse
from pathlib import Path
from calendar import monthrange
import calendar
import numpy as np
from scipy import ndimage
import multiprocessing
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from netCDF4 import Dataset as netcdf_dataset

from rss_sat_readers import read_rss_satellite_daily_xr
from rss_land_ice_maps.global_land_fraction import read_land_fraction_1440_720

#some constants to make the figures come out the same size
figsize=(16.4, 9.2)
dpi=141.6 


def ccmp_30_stream_fig_filename(
    year=2016,
    month=1,
    day=1,
    time_step=0,
    path=Path("P:/CCMP/MEaSUREs/era5_current_corrected_adj4/v2.0/plots/"),
    low_res=False,
):

    assert year >= 1993, "year out of range, year must be larger than 2015"
    assert month >= 1, "month out of range"
    assert month <= 12, "month out of range"
    num_days_in_month = monthrange(year, month)[1]
    assert day >= 1, "day out of range"
    assert day <= num_days_in_month, "day out of range"
    time_step = int(time_step)
    assert time_step >= 0, "time step out of range"
    assert time_step <= 3, "time step out of range"

    hour = time_step * 6
    if low_res:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"d{day:02d}"
            / f"CCMP_30_Wind_Analysis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}z_stream.lo.png"
        )
    else:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"d{day:02d}"
            / f"CCMP_30_Wind_Analysis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}z_stream.png"
        )
    return png_file

def ccmp_31_stream_fig_filename(
    year=2016,
    month=1,
    day=1,
    time_step=0,
    path=Path("P:/CCMP/MEaSUREs/era5_current_corrected_adj4_v31/v3.1/plots/"),
    low_res=False,
):

    assert year >= 1993, "year out of range, year must be >= than 1993"
    assert month >= 1, "month out of range"
    assert month <= 12, "month out of range"
    num_days_in_month = monthrange(year, month)[1]
    assert day >= 1, "day out of range"
    assert day <= num_days_in_month, "day out of range"
    time_step = int(time_step)
    assert time_step >= 0, "time step out of range"
    assert time_step <= 3, "time step out of range"

    hour = time_step * 6
    if low_res:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"d{day:02d}"
            / f"CCMP_31_Wind_Analysis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}z_stream.lo.png"
        )
    else:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"d{day:02d}"
            / f"CCMP_31_Wind_Analysis_{year:04d}_{month:02d}_{day:02d}_{hour:02d}z_stream.png"
        )

    return png_file


def ccmp_30_fig_filename_monthly(*,
    year : int,
    month : int,
    path=Path("P:/CCMP/MEaSUREs/era5_current_corrected_adj4/v2.0/plots/"),
    plot_type : str,
    low_res=False,
):

    assert year >= 1993, "year out of range, year must be larger than 2015"
    assert month >= 1, "month out of range"
    assert month <= 12, "month out of range"

    if low_res:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"CCMP_30_Wind_Analysis_{year:04d}_{month:02d}_{plot_type}.lo.png"
        )
    else:
        png_file = (
            path
            / f"y{year:04d}"
            / f"m{month:02d}"
            / f"CCMP_30_Wind_Analysis_{year:04d}_{month:02d}_{plot_type}.png"
        )
    return png_file

def ccmp_30_fig_filename_monthly_clim(*,
    month : int,
    path=Path("P:/CCMP/MEaSUREs/era5_current_corrected_adj4/v2.0/plots/clim"),
    plot_type : str,
    ):

    assert month >= 1, "month out of range"
    assert month <= 12, "month out of range"

    png_file = (
        path
        / f"CCMP_30_Wind_Analysis_Clim_1995-2014_{month:02d}_{plot_type}.png"
    )
    return png_file

def ccmp_30_stream_fig_filename_monthly(*,
                                        year : int,
                                        month : int,
                                        path=Path("P:/CCMP/MEaSUREs/era5_current_corrected_adj4/v2.0/plots/"),
                                        low_res=False
                                        ):

    return ccmp_30_fig_filename_monthly(year=year,month=month,path=path,plot_type='stream',low_res=low_res)


def wind_cm():
    import matplotlib.colors as mcolors

    colors1 = plt.cm.RdYlBu_r(np.linspace(0.0, 1, 170))
    colors2 = plt.cm.gist_heat_r(np.linspace(0.6, 1, 86))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)


def get_ssmi_ice_map(*, year, month, day):

    sat_to_use = "F13"
    if year < 1996:
        sat_to_use = "F11"
    if year > 2008:
        sat_to_use = "F17"
    if year > 2011:
        sat_to_use = "F18"

    ssmis = read_rss_satellite_daily_xr(
        year=year, month=month, day=day, satellite=sat_to_use, include_byte_data=True
    )
    is_ice = ssmis["byte_data"].values[2, :, :] == 252
    ice = np.zeros((720, 1440))
    ice[is_ice] = 60.0

    for ilon in range(0,1440):
        ice_column = ice[:,ilon]
        min_loc = np.min(np.where(ice > 59.0))
        if min_loc < 300:
            ice_column[0:min_loc] = 60.0

    ice[land_frac > 0.9] = np.nan
    ice = np.roll(ice, -120, axis=1)
    ice[ice < 60.0] = np.nan

    return ice


def plot_streamplot(x,y,u,v,w10,ice,year,mnth,day,time_step):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    midnorm = MidpointNormalize(vmin=0.0, v1=12.0, v2=25.0, vmax=40.0)
    central_longitude = 210.0
    cmap = wind_cm()
    central_longitude = 210.0
    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    title_str = (
        "CCMP 3.0 "
        + str(mnth).zfill(2)
        + "/"
        + str(day).zfill(2)
        + "/"
        + str(year).zfill(4)
        + " "
        + str(6 * time_step).zfill(2)
        + "z"
    )
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title_str, fontsize=18)  # title of plot

    w10_to_plot = np.ma.masked_where(~np.isfinite(w10), w10)
    map = ax.imshow(
        np.flipud(w10_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=cmap,
    )

    temp = ice[640:720]
    tempw = w10[640:720]

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    map2 = ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )

    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label("Wind Speed (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.streamplot(
        x,
        y,
        u,
        v,
        linewidth=0.5,
        density=[10, 5],
        color="tan",
        arrowsize=0.5,
        minlength=0.01,
    )
    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    png_file = ccmp_31_stream_fig_filename(
        year=year, month=mnth, day=day, time_step=time_step
    )
    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)

def plot_monthly_streamplot(x,y,u,v,w10,ice,year,mnth):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    midnorm = MidpointNormalize(vmin=0.0, v1=12.0, v2=25.0, vmax=40.0)
    central_longitude = 210.0
    cmap = wind_cm()
    central_longitude = 210.0
    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    title_str = (
        "Monthly Mean: CCMP 3.1 "
        + str(year).zfill(2)
        + "-"
        + str(mnth).zfill(2)
    )
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title_str, fontsize=18)  # title of plot

    w10_to_plot = np.ma.masked_where(~np.isfinite(w10), w10)
    map = ax.imshow(
        np.flipud(w10_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=cmap,
    )

    temp = ice[640:720]
    tempw = w10[640:720]

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    map2 = ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )

    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label("Wind Speed (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.streamplot(
        x,
        y,
        u,
        v,
        linewidth=0.5,
        density=[10, 5],
        color="tan",
        arrowsize=0.5,
        minlength=0.01,
    )
    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    png_file = ccmp_30_fig_filename_monthly(year=year,month=mnth,plot_type='stream')

    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)

def plot_monthly_clim_streamplot(x,y,u,v,w10,ice,mnth):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    midnorm = MidpointNormalize(vmin=0.0, v1=12.0, v2=25.0, vmax=40.0)
    central_longitude = 210.0
    cmap = wind_cm()
    central_longitude = 210.0
    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    title_str = f'Monthly Cimatology: CCMP 3.0 (1995-2014) Month:{mnth:02d}'

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title_str, fontsize=18)  # title of plot

    w10_to_plot = np.ma.masked_where(~np.isfinite(w10), w10)
    map = ax.imshow(
        np.flipud(w10_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=cmap,
    )

    temp = ice[640:720]
    tempw = w10[640:720]

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    map2 = ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )

    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label("Wind Speed (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.streamplot(
        x,
        y,
        u,
        v,
        linewidth=0.5,
        density=[10, 5],
        color="tan",
        arrowsize=0.5,
        minlength=0.01,
    )
    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    png_file = ccmp_30_fig_filename_monthly_clim(month=mnth,plot_type='stream')
    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)

def plot_monthly_anom_streamplot(x,y,u,v,w10,ice,year,mnth):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    midnorm = MidpointNormalize(vmin=0.0, v1=3.0, v2=6.0, vmax=10.0)
    central_longitude = 210.0
    cmap = wind_cm()
    central_longitude = 210.0
    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    title_str = (
        "Monthly Mean Vector Anomaly: CCMP 3.0 "
        + str(year).zfill(2)
        + "-"
        + str(mnth).zfill(2)
    )
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title_str, fontsize=18)  # title of plot

    w10_to_plot = np.ma.masked_where(~np.isfinite(w10), w10)
    map = ax.imshow(
        np.flipud(w10_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=cmap,
    )

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    map2 = ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )

    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label("Vector Anomaly Magnitude (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.streamplot(
        x,
        y,
        u,
        v,
        linewidth=0.5,
        density=[10, 5],
        color="tan",
        arrowsize=0.5,
        minlength=0.01,
    )
    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    png_file = ccmp_30_fig_filename_monthly(
        year=year, month=mnth,plot_type='stream_anom'
    )
    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)

def plot_clim_component_map(*,
                            w10 : np.ndarray,
                            ice : np.ndarray,
                            mnth : int,
                            start_year : int,
                            end_year : int,
                            component : str,
                            png_file : str,
                            norm,
                            cmap):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    central_longitude = 210.0
    

    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    title_str = f'CCMP 3.5 Climatological {component}, {calendar.month_name[mnth]}, {start_year}-{end_year}'
       
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title_str, fontsize=18)  # title of plot

    

    w10_to_plot = np.ma.masked_where(~np.isfinite(w10), w10)
    map = ax.imshow(
        np.flipud(w10_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=norm,
        extent=img_extent,
        cmap=cmap,
    )

    '''
    temp = ice[640:720]
    tempw = w10[640:720]

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    map2 = ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=midnorm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )
    '''

    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label(f" {component} (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    
    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)

def plot_scalar_anomaly_map(*,
                            anom : np.ndarray,
                            ice: np.ndarray,
                            title : str,
                            var='',
                            png_file : str,
                            norm,
                            cmap
                            ):


    fig =  plt.figure(
        figsize=figsize, dpi=dpi  
        ) 

    central_longitude = 210.0
    
    img_extent = [-180.0, 180.0, -90.0, 90.0]
    
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(central_longitude=central_longitude),
        position=[1.0, 1.0, 14.4, 7.2],
    )
    ax.set_title(title, fontsize=18)  # title of plot

    anom_to_plot = np.ma.masked_where(~np.isfinite(anom), anom)
    map = ax.imshow(
        np.flipud(anom_to_plot),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=norm,
        extent=img_extent,
        cmap=cmap,
    )

    ice_map = np.ma.masked_where(ice < 60.0, ice)
    ax.imshow(
        np.flipud(ice_map),
        origin="upper",
        transform=ccrs.PlateCarree(central_longitude=central_longitude),
        norm=norm,
        extent=img_extent,
        cmap=plt.cm.Greys_r,
    )
    
    cbar = fig.colorbar(
        map,
        shrink=0.70,
        orientation="vertical",
    )
    cbar.set_label(f" {var} (m/s)", size=16)
    cbar.ax.tick_params(labelsize=14)

    ax.add_feature(cfeature.LAND, facecolor="grey", zorder=10)
    
    print(png_file)
    os.makedirs(
         os.path.dirname(png_file), exist_ok=True
    )  # succeeds even if directory exists.

    fig.savefig(png_file, dpi="figure", bbox_inches="tight")
    plt.close(fig)


def spawn(func, *args):
    proc = multiprocessing.Process(target=func, args=args)
    proc.start()
    # wait until proc terminates.
    return proc


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Create Streamplot Image files from CCMP Data")
    )

    parser.add_argument(
        "start_date",
        type=datetime.date.fromisoformat,
        help="First Day to process, as YYYY-MM-DD",
    )
    parser.add_argument(
        "end_date",
        type=datetime.date.fromisoformat,
        help="Last Day to process, as YYYY-MM-DD",
    )

    parser.add_argument(
        "--overwrite", help="overwrite existing .png files", action="store_true"
    )

    args = parser.parse_args()

    START_DAY = args.start_date
    END_DAY = args.end_date
    date = START_DAY

    cmap = wind_cm()
    central_longitude = 210.0
    land_frac = read_land_fraction_1440_720()
    land_frac = ndimage.uniform_filter(land_frac, size=5, mode="wrap")
    img_extent = [-180.0, 180.0, -90.0, 90.0]

    

    date = START_DAY
    while date <= END_DAY:
        year = date.year
        mnth = date.month
        day = date.day

        try:
            ice_temp = get_ssmi_ice_map(year=year, month=mnth, day=day)
            ice = ice_temp
        except:
            print("Ice mask not constructed")
            print("Using Ice from previous day")

        for time_step in range(0, 4):

            png_file_to_test = Path(
                ccmp_30_stream_fig_filename(
                    year=year, month=mnth, day=day, time_step=time_step
                )
            )
            if png_file_to_test.is_file():
                if args.overwrite:
                    print("png file already exists, overwriting")
                else:
                    print("png file already exists, skipping")
                    continue

            try:
                ccmp_wind = read_ccmp_30(
                    year=year, month=mnth, day=day, time_step=time_step
                )
            except FileNotFoundError:
                print("File not found, skipping")
                continue

            u = ccmp_wind["u10"]

            u[land_frac > 0.95] = np.nan
            u = np.roll(u, -120, axis=1)
            u[np.isfinite(ice)] = np.nan

            v = ccmp_wind["v10"]

            v[land_frac > 0.95] = np.nan
            v = np.roll(v, -120, axis=1)
            v[np.isfinite(ice)] = np.nan

            w10 = (u**2 + v**2) ** 0.5
            x = 0.0 + 0.125 + 0.25 * np.arange(0.0, 1440.0)
            y = 0.125 + 0.25 * np.arange(0.0, 720.0) - 90.0
            x[x > 180.0] = x[x > 180.0] - 360.0

            vmin = 0.0
            vmax = 25.0
            img_extent = [-180.0, 180.0, -90.0, 90.0]

            u = np.roll(u, 720, axis=1)
            v = np.roll(v, 720, axis=1)

            spawn(plot_streamplot,x,y,u,v,w10,ice,year,mnth,day,time_step)
            
        date += datetime.timedelta(days=1)
