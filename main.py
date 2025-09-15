# =========================================
# Tempo API v1.0.0
# =========================================

# ===============================
# IMPORTS AND DEPENDENCIES
# ===============================

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from ecmwf.opendata import Client
import xarray as xr
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import os
import json
import glob
import re
import time
import matplotlib.pyplot as plt
import geojsoncontour
from shapely.geometry import shape, LineString
from scipy import ndimage
import shapely

# ===============================
# APPLICATION INITIALIZATION
# ===============================

app = FastAPI(
    title="Tempo API",
    version="1.0.0",
    description="""
Tempo API provides weather data visualization and analysis endpoints, including:
- WebP weather maps for various meteorological parameters
- GeoJSON isolines for mean sea level pressure
- Data statistics and metadata endpoints

All endpoints are designed for fast, automated weather data delivery. Data source: ECMWF OpenData (European Centre for Medium-Range Weather Forecasts).

Endpoints are grouped by weather theme for easier navigation.
"""
)

# Load weather themes configuration
with open("themes.json") as f:
    themes = json.load(f)

# Ensure required directories exist
os.makedirs("static/grib", exist_ok=True)

# ===============================
# UTILITY FUNCTIONS
# ===============================


def clean_old_cache(param: str, cache_dir: str = "static", max_age_hours: int = 24):
    now = time.time()
    pattern = os.path.join(cache_dir, f"{param}_*_*.json")

    for file_path in glob.glob(pattern):
        try:
            mtime = os.path.getmtime(file_path)
            age_hours = (now - mtime) / 3600.0

            if age_hours > max_age_hours:
                os.remove(file_path)
                print(f"[CACHE] Removed old cache file: {file_path}")
        except Exception as e:
            print(f"[CACHE] Error removing {file_path}: {e}")


def parse_time_param(time_str: str) -> datetime:
    try:
        ts = datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid time format: use YYYYMMDDHH")

    # Validate timestamp is within allowed range (-24h to +24h from now)
    now = datetime.utcnow()
    if not (now - timedelta(hours=24) <= ts <= now + timedelta(hours=24)):
        raise HTTPException(
            status_code=400, detail="Timestamp out of allowed range (-24h to +24h)")

    return ts

# ===============================
# GRIB DATA PROCESSING
# ===============================


def download_grib(file_path: str, param: str, force: bool = False) -> xr.DataArray:
    # Extract timestamp from file path
    match = re.search(r"(\d{10})", file_path)
    time_param = match.group(
        1) if match else datetime.utcnow().strftime("%Y%m%d%H")
    cache_path = os.path.join("static", "grib", f"{param}_{time_param}.grib2")

    # Use cached file if available and not forcing refresh
    if os.path.exists(cache_path) and not force:
        print(f"[CACHE] Using cached GRIB file: {cache_path}")
        ds = xr.open_dataset(cache_path, engine="cfgrib")
        data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]

        # Ensure time dimension exists
        if "time" not in data.dims:
            data = data.expand_dims("time")

        print(f"[CACHE] GRIB file loaded from cache.")
        return data.isel(time=0)

    # Download fresh data from ECMWF
    print(f"[DOWNLOAD] Downloading GRIB file from ECMWF: {cache_path}")

    # Remove existing index file to avoid conflicts
    idx_file = cache_path + ".idx"
    if os.path.exists(idx_file):
        os.remove(idx_file)

    # Download using ECMWF OpenData client
    client = Client(source="ecmwf")
    client.retrieve(
        time=0,
        stream="oper",
        type="fc",
        step=24,
        param=param,
        target=cache_path
    )

    print(f"[DOWNLOAD] Download complete: {cache_path}")

    # Load and process the downloaded data
    ds = xr.open_dataset(cache_path, engine="cfgrib")
    data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]

    # Ensure time dimension exists
    if "time" not in data.dims:
        data = data.expand_dims("time")

    print(f"[PROCESS] GRIB file loaded and processed.")
    return data.isel(time=0)

# ===============================
# IMAGE PROCESSING AND VISUALIZATION
# ===============================


def apply_palette_to_data_vectorized(data_array: np.ndarray, palette: list) -> np.ndarray:
    # Sort palette by data values and extract components
    sorted_palette = sorted(palette, key=lambda x: x[0])
    levels, colors = zip(*sorted_palette)
    levels = np.array(levels)
    colors = np.array(colors)[:, :3]

    # Initialize RGB output array
    rgb_array = np.zeros((*data_array.shape, 3), dtype=np.uint8)

    # Clip data to palette range
    data_clipped = np.clip(data_array, levels[0], levels[-1])

    # Find interpolation indices for each data point
    idx = np.searchsorted(levels, data_clipped) - 1
    idx[idx < 0] = 0

    # Get boundary values and colors for interpolation
    v1 = levels[idx]
    v2 = levels[idx + 1]
    c1 = colors[idx]
    c2 = colors[idx + 1]

    # Linear interpolation between colors
    ratio = np.expand_dims((data_clipped - v1) / (v2 - v1 + 1e-8), axis=-1)
    rgb_array = (c1 + ratio * (c2 - c1)).astype(np.uint8)

    # Handle NaN values with gray color
    rgb_array[np.isnan(data_array)] = [128, 128, 128]

    return rgb_array

# ===============================
# CONTOUR LINE GENERATION
# ===============================


def generate_isolines_geojson(array: np.ndarray, lat_coords=None, lon_coords=None,
                              interval=2.0, ndigits=3, unit='m') -> dict:
    interval = float(interval)
    if interval <= 0:
        raise ValueError("Interval must be positive.")

    # Validate input data
    valid_data = array[~np.isnan(array)]
    if len(valid_data) == 0:
        print("[ISOLINES] No valid data available for isoline generation.")
        return {"type": "FeatureCollection", "features": []}

    # Skip uniform data (no contours possible)
    if np.all(valid_data == valid_data[0]):
        print(
            f"[ISOLINES] Uniform data ({valid_data[0]}), no isolines to generate.")
        return {"type": "FeatureCollection", "features": []}

    # Apply Gaussian smoothing for cleaner meteorological contours

    # Sigma values for smoothing
    sigma = [3.0, 3.0]  # [lat_sigma, lon_sigma]
    print(f"[ISOLINES] Applying Gaussian smoothing with sigma={sigma}")

    # Apply Gaussian filter with constant boundary conditions
    array = ndimage.gaussian_filter(array, sigma, mode='constant', cval=np.nan)
    print("[ISOLINES] Gaussian smoothing applied successfully")

    # Calculate contour levels
    data_min, data_max = np.floor(
        np.min(valid_data)), np.ceil(np.max(valid_data))
    print(
        f"[ISOLINES] Data range: {data_min} to {data_max}, Interval: {interval}")

    levels = np.arange(
        np.floor(data_min / interval) * interval,
        np.ceil(data_max / interval) * interval + interval,
        interval
    )
    print(f"[ISOLINES] Contour levels: {levels}")

    # Set up coordinate system
    height, width = array.shape
    if lat_coords is None:
        lat_coords = np.linspace(-90, 90, height)
    if lon_coords is None:
        lon_coords = np.linspace(-180, 180, width)

    # Create coordinate meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # Generate contours using matplotlib
    figure = plt.figure(figsize=(12, 8))
    ax = figure.add_subplot(111)

    try:
        # Create filled contour plot
        contourf = ax.contourf(lon_grid, lat_grid, array,
                               levels=levels, extend='both')

        # Convert matplotlib contours to GeoJSON
        geojson_str = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            ndigits=ndigits,
            unit=unit,
            stroke_width=1,
            fill_opacity=0
        )

    except Exception as e:
        print(f"[ISOLINES] Error generating contours: {e}")
        return {"type": "FeatureCollection", "features": []}
    finally:
        plt.close(figure)

    # Process GeoJSON to extract line features
    try:
        geo = json.loads(geojson_str)
        line_features = []

        for feature in geo["features"]:
            geom = shape(feature["geometry"])
            level_value = feature["properties"].get("level", None)

            # Convert polygons and multipolygons to line strings
            if geom.geom_type == "Polygon":
                line = LineString(geom.exterior.coords)
                line = shapely.simplify(
                    line, tolerance=0.05, preserve_topology=True)
                line_features.append(create_line_feature(line, level_value))

            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    line = LineString(poly.exterior.coords)
                    line = shapely.simplify(
                        line, tolerance=0.05, preserve_topology=True)
                    line_features.append(
                        create_line_feature(line, level_value))

            elif geom.geom_type == "LineString":
                line = shapely.simplify(
                    geom, tolerance=0.05, preserve_topology=True)
                line_features.append(create_line_feature(line, level_value))
            else:
                print(
                    f"[ISOLINES] Skipping unsupported geometry type: {geom.geom_type}")

        print(f"[ISOLINES] Generated {len(line_features)} isoline features")
        return {"type": "FeatureCollection", "features": line_features}

    except Exception as e:
        print(f"[ISOLINES] Error processing GeoJSON: {e}")
        return {"type": "FeatureCollection", "features": []}


def create_line_feature(line_geom, level_value):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(x), float(y)] for x, y in line_geom.coords]
        },
        "properties": {
            "level": level_value,
        }
    }

# ===============================
# ENDPOINT FACTORY FUNCTIONS
# ===============================


def create_webp_endpoint(file_path, param, palette):
    def endpoint(time_param: str, force: bool = False):
        """
        Returns a weather map as a WebP image for the given theme and timestamp.
        If the theme is wind, combines U and V components into a colorized wind speed map.
        Parameters:
            time_param (str): UTC timestamp in YYYYMMDDHH format
            force (bool): If True, forces regeneration of the image
        Returns:
            WebP image file
        """
        parse_time_param(time_param)
        # VECTOR theme: wind
        if isinstance(param, list) and len(param) == 2:
            webp_path = os.path.join("static", f"wind_{time_param}_rgb.webp")
            if os.path.exists(webp_path) and not force:
                print(f"[CACHE] Using cached WebP: {webp_path}")
                return FileResponse(webp_path, media_type="image/webp")

            print(f"[PROCESS] Generating new VECTOR WebP: {webp_path}")
            # Download U and V
            u_data = download_grib(file_path[0], param[0], force)
            v_data = download_grib(file_path[1], param[1], force)
            u = np.nan_to_num(u_data.values)
            v = np.nan_to_num(v_data.values)
            mod = np.sqrt(u**2 + v**2)
            # Apply palette to wind speed (modulus)
            rgb_array = apply_palette_to_data_vectorized(mod, palette)
            img = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
            img.save(webp_path, format="WEBP", quality=95, method=6)
            print(f"[PROCESS] VECTOR WebP file generated: {webp_path}")
            return FileResponse(webp_path, media_type="image/webp")
        else:
            webp_path = os.path.join(
                "static", f"{param}_{time_param}_rgb.webp")
            if os.path.exists(webp_path) and not force:
                print(f"[CACHE] Using cached WebP: {webp_path}")
                return FileResponse(webp_path, media_type="image/webp")

            print(f"[PROCESS] Generating new WebP: {webp_path}")
            data = download_grib(file_path, param, force)
            array = np.nan_to_num(data.values, nan=np.nanmean(data.values))
            rgb_array = apply_palette_to_data_vectorized(array, palette)
            img = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
            img.save(webp_path, format="WEBP", quality=95, method=6)
            print(f"[PROCESS] WebP file generated: {webp_path}")
            return FileResponse(webp_path, media_type="image/webp")
    return endpoint

def create_isolines_endpoint(file_path: str, param: str):
    def endpoint(time_param: str, force: bool = False, background_tasks: BackgroundTasks = None, request: Request = None):
        """
        Returns GeoJSON isolines for mean sea level pressure for the given timestamp.
        If accessed from Swagger UI, returns only a sample (10 features) for preview.
        Parameters:
            time_param (str): UTC timestamp in YYYYMMDDHH format
            force (bool): If True, forces regeneration of the isolines
        Returns:
            GeoJSON file or sample dict
        """
        parse_time_param(time_param)
        data = download_grib(file_path, param, force)
        array = np.nan_to_num(data.values, nan=np.nanmean(data.values))

        valid_values = array[~np.isnan(array)]
        data_min = float(np.min(valid_values))
        data_max = float(np.max(valid_values))
        suggested_interval = max(1, round((data_max - data_min) / 50, 1))

        geojson_path = os.path.join(
            "static", f"{param}_{time_param}_{suggested_interval}_isolines.json")

        if background_tasks:
            background_tasks.add_task(clean_old_cache, param)

        # Verificar se é Swagger UI
        is_swagger = request and "/docs" in request.headers.get("referer", "")

        if os.path.exists(geojson_path) and not force:
            if is_swagger:
                with open(geojson_path, 'r') as f:
                    full_data = json.load(f)
                return {
                    "type": "FeatureCollection",
                    "features": full_data["features"][:10],
                    "note": f"Swagger preview: 10 of {len(full_data['features'])} features"
                }
            return FileResponse(geojson_path, media_type="application/json")

        # Gerar novos isolines
        lat_coords = data.coords.get('latitude', data.coords.get('lat'))
        lon_coords = data.coords.get('longitude', data.coords.get('lon'))
        lat_values = lat_coords.values if lat_coords is not None else None
        lon_values = lon_coords.values if lon_coords is not None else None

        geojson_data = generate_isolines_geojson(
            array, lat_values, lon_values, suggested_interval)

        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)

        if is_swagger:
            return {
                "type": "FeatureCollection",
                "features": geojson_data["features"][:10],
                "note": f"Swagger preview: 10 of {len(geojson_data['features'])} features"
            }

        return FileResponse(geojson_path, media_type="application/json")

    return endpoint


def create_info_endpoint(file_path: str, param: str, units: str):
    def endpoint(time_param: str, force: bool = False):
        """
        Returns metadata and statistics for the selected weather parameter and timestamp.
        Includes value range, mean, standard deviation, spatial bounds, and grid resolution.
        Parameters:
            time_param (str): UTC timestamp in YYYYMMDDHH format
            force (bool): If True, forces data refresh
        Returns:
            JSON dict with statistics and metadata
        """
        # Validate timestamp
        parse_time_param(time_param)

        # Load weather data
        data = download_grib(file_path, param, force)

        # Extract coordinate information
        lat_coords = data.coords.get('latitude', data.coords.get('lat'))
        lon_coords = data.coords.get('longitude', data.coords.get('lon'))

        # Calculate spatial bounds and resolution
        bounds = [-180, -90, 180, 90]  # Default global bounds
        resolution = {"lat": 0.25, "lon": 0.25}  # Default ECMWF resolution

        if lat_coords is not None and lon_coords is not None:
            lat_values, lon_values = lat_coords.values, lon_coords.values
            bounds = [
                float(lon_values.min()), float(lat_values.min()),
                float(lon_values.max()), float(lat_values.max())
            ]

            # Calculate actual resolution
            if len(lat_values) > 1:
                resolution["lat"] = float(abs(lat_values[1] - lat_values[0]))
            if len(lon_values) > 1:
                resolution["lon"] = float(abs(lon_values[1] - lon_values[0]))

        # Calculate data statistics
        values = data.values
        valid_values = values[~np.isnan(values)]
        data_min, data_max = float(
            np.min(valid_values)), float(np.max(valid_values))

        return {
            "parameter": param,
            "units": units,
            "timestamp": time_param,
            "data_statistics": {
                "min_value": data_min,
                "max_value": data_max,
                "mean_value": float(np.mean(valid_values)),
                "std_deviation": float(np.std(valid_values))
            },
            "spatial_info": {
                "data_shape": list(data.shape),
                "bounds": bounds,
                "resolution": resolution
            },
        }

    return endpoint

# ===============================
# DYNAMIC ENDPOINT REGISTRATION
# ===============================

# Register endpoints for each weather theme

for theme, cfg in themes.items():
    # WebP weather map endpoint
    app.get(
        f"/{theme}/{{time_param}}/data.webp",
        tags=[theme.replace('_', ' ').title()]
    )(
        create_webp_endpoint(cfg["file"], cfg["variable"], cfg["palette"])
    )

    # GeoJSON isolines endpoint only for mean sea level pressure
    if theme == "mean_sea_level_pressure":
        app.get(
            f"/{theme}/{{time_param}}/isolines.geojson",
            tags=[theme.replace('_', ' ').title()]
        )(
            create_isolines_endpoint(cfg["file"], cfg["variable"])
        )

    # Data information endpoint
    app.get(
        f"/{theme}/{{time_param}}/info",
        tags=[theme.replace('_', ' ').title()]
    )(
        create_info_endpoint(cfg["file"], cfg["variable"], cfg["units"])
    )

# ===============================
# STATIC AND ROOT ENDPOINTS
# ===============================


@app.get("/themes.json", tags=["Static"], response_class=FileResponse)
def get_themes_json():
    """
    Returns the weather themes configuration file in JSON format.
    Useful for discovering available weather map types and their parameters.
    """
    return FileResponse("themes.json", media_type="application/json")


@app.get("/", tags=["Static"])
def redirect_to_docs():
    """
    Redirects the root URL to the interactive API documentation (Swagger UI).
    """
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Static"])
def health_check():
    """
    Health check endpoint for monitoring API status and cache usage.
    Returns basic status, current UTC timestamp, cache directory, and number of loaded themes.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_directory": "static/",
        "themes_loaded": len(themes)
    }
