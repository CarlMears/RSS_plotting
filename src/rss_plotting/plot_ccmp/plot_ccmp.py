import os
import datetime
import argparse
from pathlib import Path
from calendar import monthrange
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


def read_ccmp_30(
    *,
    year,
    month,
    day,
    time_step,
    verbose=False,
    path="P:/CCMP/MEaSUREs/era5_current_corrected_adj4/v2.0/netcdf/",
):

    assert year >= 1979, "year out of range, year must be larger than 2015"
    assert month >= 1, "month out of range"
    assert month <= 12, "month out of range"
    num_days_in_month = monthrange(year, month)[1]
    assert day >= 1, "day out of range"
    assert day <= num_days_in_month, "day out of range"
    assert time_step >= 0, "time step out of range"
    assert time_step <= 3, "time step out of range"

    path = Path(path)
    nc_file = (
        path
        / f"y{year:04d}"
        / f"m{month:02d}"
        / f"CCMP_Wind_Analysis_{year:04d}{month:02d}{day:02d}_V03.0beta2_L3.0_RSS.nc"
    )
    if verbose:
        print(f"Reading {nc_file}")

    with netcdf_dataset(nc_file) as dataset:
        u10 = dataset.variables["uwnd"][:, :, time_step]
        v10 = dataset.variables["vwnd"][:, :, time_step]
        lons = dataset.variables["longitude"][:]
        lats = dataset.variables["latitude"][:]

    ccmp_wind = dict(lats=lats, lons=lons, u10=u10, v10=v10)

    return ccmp_wind


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
    ice[land_frac > 0.9] = np.nan
    ice = np.roll(ice, -120, axis=1)
    ice[ice < 60.0] = np.nan

    return ice


def plot_streamplot(x,y,u,v,w10,ice,year,mnth,day,time_step):

    # this is kludgy way of making the map part of the come out at 1440x720
    fig =  plt.figure(
        figsize=(16.4, 9.2), dpi=141.6  
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
    png_file = ccmp_30_stream_fig_filename(
        year=year, month=mnth, day=day, time_step=time_step
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
        figsize=(16.4, 9.2), dpi=141.6  
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


def spawn(func, *args):
    proc = multiprocessing.Process(target=func, args=args)
    proc.start()
    # wait until proc terminates.
    proc.join()


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
