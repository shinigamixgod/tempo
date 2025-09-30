"""
Tempo API v1.0.0
Weather data visualization and analysis API using ECMWF OpenData.
Provides WebP maps, GeoJSON isolines, byte arrays for WebGL, and metadata endpoints.
"""

from fastapi import HTTPException
import shapely
from scipy.ndimage import zoom
from scipy import ndimage
from shapely.geometry import shape, LineString
import geojsoncontour
import matplotlib.pyplot as plt
import time
import glob
import json
import os
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import xarray as xr
from ecmwf.opendata import Client
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='cfgrib')

# =========================================
# IMPORTS AND DEPENDENCIES
# =========================================


# =========================================
# APPLICATION INITIALIZATION
# =========================================

app = FastAPI(
    title="Tempo API",
    version="1.0.0",
    description="""
Tempo API provides weather data visualization and analysis endpoints:
- WebP weather maps for various meteorological parameters
- GeoJSON isolines for mean sea level pressure
- Byte arrays for WebGL particle systems
- Data statistics and metadata endpoints

Data source: ECMWF OpenData (European Centre for Medium-Range Weather Forecasts).
"""
)

# Load weather themes configuration
with open("themes.json") as f:
    themes = json.load(f)

# Ensure required directories exist
os.makedirs("static/grib", exist_ok=True)

# =========================================
# UTILITY FUNCTIONS
# =========================================


def clean_old_cache(param: str, cache_dir: str = "static", max_age_hours: int = 24):
    """
    Remove cached files older than max_age_hours for a given parameter.

    Args:
        param: Weather parameter name
        cache_dir: Directory containing cached files
        max_age_hours: Maximum age in hours before deletion
    """
    now = time.time()
    pattern = os.path.join(cache_dir, f"{param}_*_*.json")

    for file_path in glob.glob(pattern):
        try:
            mtime = os.path.getmtime(file_path)
            age_hours = (now - mtime) / 3600.0

            if age_hours > max_age_hours:
                os.remove(file_path)
        except Exception:
            pass


def parse_time_param(time_str: str) -> datetime:
    """
    Parse and validate time parameter string.

    Args:
        time_str: Time string in format YYYYMMDDHH

    Returns:
        Validated datetime object

    Raises:
        HTTPException: If format invalid or timestamp out of range (-120h to +120h)
    """
    try:
        ts = datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid time format: use YYYYMMDDHH"
        )

    # Allow range: -120h to +120h from now
    now = datetime.utcnow()
    if not (now - timedelta(hours=120) <= ts <= now + timedelta(hours=120)):
        raise HTTPException(
            status_code=400,
            detail="Timestamp out of allowed range (-120h to +120h)"
        )

    return ts


# =========================================
# GRIB DATA PROCESSING
# =========================================


def download_grib(param: str, use_cache: bool = True, time_param: str = None) -> xr.DataArray:
    """Download GRIB data with automatic fallback to available cycles."""
    if time_param is None:
        time_param = datetime.utcnow().strftime("%Y%m%d%H")
    
    dt = datetime.strptime(time_param, "%Y%m%d%H")
    valid_cycles = [0, 6, 12, 18]
    now = datetime.utcnow()
    
    # Determine initial cycle and step
    cycle_hour = max([h for h in valid_cycles if h <= dt.hour], default=0)
    cycle_dt = dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    step = int((dt - cycle_dt).total_seconds() // 3600)
    
    # Normalize step to multiples of 3
    if step % 3 != 0:
        step = step - (step % 3)
    
    # Build cache path
    cycle_str = cycle_dt.strftime('%Y%m%d')
    cache_path = os.path.join("static", "grib", f"{param}_{cycle_str}{cycle_hour:02d}_{step}.grib2")
    
    print(f"[GRIB] Requested: time={time_param}, cycle={cycle_str} {cycle_hour:02d}h, step={step}h")
    
    # Check cache first
    if os.path.exists(cache_path) and use_cache:
        print(f"[GRIB] Using cached file: {cache_path}")
        ds = xr.open_dataset(cache_path, engine="cfgrib")
        data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
        if "time" not in data.dims:
            data = data.expand_dims("time")
        return data.isel(time=0)
    
    # Build list of cycles to try (current and previous cycles)
    cycles_to_try = []
    
    # For requested time, try current cycle with different steps
    for s in range(step, -1, -3):
        cycles_to_try.append((cycle_dt, cycle_hour, s))
    
    # Try previous cycles on the same day
    for prev_hour in reversed([h for h in valid_cycles if h < cycle_hour]):
        prev_cycle = dt.replace(hour=prev_hour, minute=0, second=0, microsecond=0)
        prev_step = int((dt - prev_cycle).total_seconds() // 3600)
        if prev_step % 3 != 0:
            prev_step = prev_step - (prev_step % 3)
        if 0 <= prev_step <= 240:
            cycles_to_try.append((prev_cycle, prev_hour, prev_step))
    
    # Try previous day's cycles if needed
    if dt.hour < 6:
        yesterday = dt - timedelta(days=1)
        for prev_hour in reversed(valid_cycles):
            prev_cycle = yesterday.replace(hour=prev_hour, minute=0, second=0, microsecond=0)
            prev_step = int((dt - prev_cycle).total_seconds() // 3600)
            if prev_step % 3 != 0:
                prev_step = prev_step - (prev_step % 3)
            if 0 <= prev_step <= 240:
                cycles_to_try.append((prev_cycle, prev_hour, prev_step))
    
    # Try each cycle/step combination
    client = Client(source="ecmwf")
    
    for try_cycle, try_hour, try_step in cycles_to_try:
        try_cache_path = os.path.join(
            "static", "grib", 
            f"{param}_{try_cycle.strftime('%Y%m%d')}{try_hour:02d}_{try_step}.grib2"
        )
        
        # Check if this cycle is already cached
        if os.path.exists(try_cache_path):
            print(f"[GRIB] Found cached alternative: cycle={try_cycle.strftime('%Y%m%d %H')}h, step={try_step}h")
            ds = xr.open_dataset(try_cache_path, engine="cfgrib")
            data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
            if "time" not in data.dims:
                data = data.expand_dims("time")
            return data.isel(time=0)
        
        # Try to download
        try:
            print(f"[GRIB] Trying: cycle={try_cycle.strftime('%Y%m%d %H')}h, step={try_step}h")
            
            # Remove old index file if exists
            idx_file = try_cache_path + ".idx"
            if os.path.exists(idx_file):
                os.remove(idx_file)
            
            client.retrieve(
                date=try_cycle.strftime('%Y%m%d'),
                time=try_hour,
                type="fc",
                stream="oper",
                step=try_step,
                param=param,
                target=try_cache_path
            )
            
            print(f"[GRIB] Success! Downloaded: cycle={try_cycle.strftime('%Y%m%d %H')}h, step={try_step}h")
            
            ds = xr.open_dataset(try_cache_path, engine="cfgrib")
            data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
            if "time" not in data.dims:
                data = data.expand_dims("time")
            return data.isel(time=0)
            
        except Exception as e:
            import requests
            if hasattr(e, 'response') and isinstance(e.response, requests.Response) and e.response.status_code == 404:
                print(f"[GRIB] Not found: cycle={try_cycle.strftime('%Y%m%d %H')}h, step={try_step}h (404)")
                continue
            else:
                print(f"[GRIB] Error: {str(e)}")
                # For non-404 errors, continue trying other cycles
                continue
    
    # If we've tried everything and nothing worked
    raise HTTPException(
        status_code=404, 
        detail=f"No ECMWF data available for {param} near {time_param}. Tried {len(cycles_to_try)} cycle/step combinations."
    )

    # Get timestamp from time_param (always from request)
    # file_path is only the base name, not timestamp
    # time_param is always passed by endpoints
    if time_param is None:
        time_param = datetime.utcnow().strftime("%Y%m%d%H")
    dt = datetime.strptime(time_param, "%Y%m%d%H")
    valid_cycles = [0, 6, 12, 18]
    cycle_hour = max([h for h in valid_cycles if h <= dt.hour], default=None)
    if cycle_hour is None:
        cycle_hour = valid_cycles[0]
    cycle_dt = dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    step = int((dt - cycle_dt).total_seconds() // 3600)
    time = cycle_hour

    # If step is not valid, use previous cycle and step 0
    if step not in valid_cycles:
        idx = valid_cycles.index(cycle_hour)
        if idx > 0:
            cycle_hour = valid_cycles[idx-1]
            cycle_dt = dt.replace(
                hour=cycle_hour, minute=0, second=0, microsecond=0)
            step = int((dt - cycle_dt).total_seconds() // 3600)
            time = cycle_hour
            pass
        else:
            pass
            step = 0
            cycle_dt = dt.replace(
                hour=valid_cycles[0], minute=0, second=0, microsecond=0)
            time = valid_cycles[0]


    # O cache é sempre por ciclo/step do timestamp solicitado
    cycle_str = cycle_dt.strftime('%Y%m%d')
    cache_path = os.path.join(
        "static", "grib", f"{param}_{cycle_str}{time:02d}_{step}.grib2")

    # Diagnostic print for debugging GRIB retrieval
    print(f"[GRIB-LEGACY] param={param}, time_param={time_param}, cycle={cycle_str} hour={time}, step={step}, cache_path={cache_path}")

    # Requested param, time_param, file_path, cache_path, force

    # Use cache file if available and not forcing download
    if os.path.exists(cache_path) and not use_cache:
        # Using cached GRIB file
        ds = xr.open_dataset(cache_path, engine="cfgrib")
        data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
        if "time" not in data.dims:
            data = data.expand_dims("time")
        # GRIB file loaded from cache
        return data.isel(time=0)

    # Calculate time and step for ECMWF client
    dt = datetime.strptime(time_param, "%Y%m%d%H")
    valid_cycles = [0, 6, 12, 18]
    cycle_hour = max([h for h in valid_cycles if h <= dt.hour], default=None)
    if cycle_hour is None:
        raise HTTPException(
            status_code=400, detail=f"No valid ECMWF cycle for hour {dt.hour}")
    cycle_dt = dt.replace(hour=cycle_hour)
    step = int((dt - cycle_dt).total_seconds() // 3600)
    time = cycle_hour

    # If step is not valid, use previous cycle and step 0
    if step not in valid_cycles:
        idx = valid_cycles.index(cycle_hour)
        if idx > 0:
            cycle_hour = valid_cycles[idx-1]
            cycle_dt = dt.replace(hour=cycle_hour)
            step = int((dt - cycle_dt).total_seconds() // 3600)
            time = cycle_hour
        else:
            step = 0
            cycle_dt = dt.replace(hour=valid_cycles[0])
            time = valid_cycles[0]

    # Validations
    now = datetime.utcnow()
    ciclo_futuro = dt > now
    if step < 0:
        step = 0
    # ECMWF only accepts steps multiple of 3
    if step % 3 != 0:
        step = step - (step % 3)
    if step > 240:
        raise HTTPException(
            status_code=400, detail=f"Step {step}h out of ECMWF range (max 240h)")

    def try_retrieve(step_value):
        try:
            if ciclo_futuro:
                # If future timestamp, get latest ECMWF cycle and adjust step
                client = Client(source="ecmwf")
                result = client.retrieve(
                    time=time,
                    type="fc",
                    stream="oper",
                    step=0,
                    param=param,
                    target=cache_path
                )
                print(f"[GRIB-LEGACY] Retrieved latest cycle for future timestamp: original cycle={cycle_dt}, original step={step}, requested cycle hour={time}")
                ciclo_real = result.datetime
                step_corrigido = int((dt - ciclo_real).total_seconds() // 3600)
                if step_corrigido < 0:
                    return False
                client.retrieve(
                    time=time,
                    type="fc",
                    stream="oper",
                    step=step_corrigido,
                    param=param,
                    target=cache_path
                )
                print(f"[GRIB-LEGACY] Future cycle adjusted: original cycle={cycle_dt}, original step={step}, actual cycle={ciclo_real}, adjusted step={step_corrigido}")
            else:
                idx_file = cache_path + ".idx"
                if os.path.exists(idx_file):
                    os.remove(idx_file)
                client = Client(source="ecmwf")
                client.retrieve(
                    date=cycle_dt.strftime('%Y%m%d'),
                    time=time,
                    type="fc",
                    stream="oper",
                    step=step_value,
                    param=param,
                    target=cache_path
                )
                print(f"[GRIB-LEGACY] Retrieved past cycle: {cycle_dt}, step={step_value}")
            return True
        except Exception as e:
            import requests
            if hasattr(e, 'response') and isinstance(e.response, requests.Response) and e.response.status_code == 404:
                return False
            else:
                raise

    # Try requested step, then previous valid steps
    fallback_steps = [step] + [s for s in range(step-3, -1, -3) if s >= 0]
    found = False
    for s in fallback_steps:
        if try_retrieve(s):
            found = True
            break
    if not found:
        raise HTTPException(
            status_code=404, detail=f"No ECMWF data for cycle={cycle_dt}, param={param}, steps={fallback_steps}")
    ds = xr.open_dataset(cache_path, engine="cfgrib")
    data = ds[param] if param in ds else ds[list(ds.data_vars)[0]]
    if "time" not in data.dims:
        data = data.expand_dims("time")
    return data.isel(time=0)


# =========================================
# IMAGE PROCESSING AND VISUALIZATION
# =========================================

def apply_palette_to_data_vectorized(data_array: np.ndarray, palette: list) -> np.ndarray:
    """
    Apply color palette to data array using vectorized interpolation.

    Args:
        data_array: 2D numpy array with data values
        palette: List of [value, [R, G, B]] color stops

    Returns:
        3D RGB array with interpolated colors
    """
    # Sort palette by value and extract levels and colors
    sorted_palette = sorted(palette, key=lambda x: x[0])
    levels, colors = zip(*sorted_palette)
    levels = np.array(levels)
    colors = np.array(colors)[:, :3]  # RGB only

    # Create RGB output array
    rgb_array = np.zeros((*data_array.shape, 3), dtype=np.uint8)

    # Clip data to palette range
    data_clipped = np.clip(data_array, levels[0], levels[-1])

    # Find color index for each data point
    idx = np.searchsorted(levels, data_clipped) - 1
    idx[idx < 0] = 0

    # Get color boundaries for interpolation
    v1 = levels[idx]
    v2 = levels[idx + 1]
    c1 = colors[idx]
    c2 = colors[idx + 1]

    # Linear interpolation between colors
    ratio = np.expand_dims((data_clipped - v1) / (v2 - v1 + 1e-8), axis=-1)
    rgb_array = (c1 + ratio * (c2 - c1)).astype(np.uint8)

    # Set NaN values to gray
    rgb_array[np.isnan(data_array)] = [128, 128, 128]

    return rgb_array


def generate_isolines_geojson(array: np.ndarray, lat_coords=None, lon_coords=None,
                              interval=2.0, ndigits=3, unit='hPa') -> dict:
    """
    Generate GeoJSON isolines from 2D data array.

    Args:
        array: 2D numpy array with data values
        lat_coords: Latitude coordinate values
        lon_coords: Longitude coordinate values
        interval: Contour interval
        ndigits: Number of decimal places for coordinates
        unit: Unit of measurement for properties

    Returns:
        GeoJSON FeatureCollection with LineString features
    """
    interval = float(interval)
    if interval <= 0:
        raise ValueError("Interval must be positive.")

    # Check for valid data
    valid_data = array[~np.isnan(array)]
    if len(valid_data) == 0:
        return {"type": "FeatureCollection", "features": []}

    # Skip if all data is uniform
    if np.all(valid_data == valid_data[0]):
        return {"type": "FeatureCollection", "features": []}

    # Apply Gaussian smoothing for better contours
    sigma = [3.0, 3.0]
    array = ndimage.gaussian_filter(array, sigma, mode='constant', cval=np.nan)

    # Calculate contour levels
    data_min, data_max = np.floor(
        np.min(valid_data)), np.ceil(np.max(valid_data))
    levels = np.arange(
        np.floor(data_min / interval) * interval,
        np.ceil(data_max / interval) * interval + interval,
        interval
    )

    # Set up coordinate grids
    height, width = array.shape
    if lat_coords is None:
        lat_coords = np.linspace(-90, 90, height)
    if lon_coords is None:
        lon_coords = np.linspace(-180, 180, width)

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # Generate contours using matplotlib
    figure = plt.figure(figsize=(12, 8))
    ax = figure.add_subplot(111)

    try:
        contourf = ax.contourf(lon_grid, lat_grid, array,
                               levels=levels, extend='both')

        # Convert to GeoJSON
        geojson_str = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            ndigits=ndigits,
            unit=unit,
            stroke_width=1,
            fill_opacity=0
        )
    except Exception:
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

            # Convert polygon boundaries to line strings
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

        return {"type": "FeatureCollection", "features": line_features}

    except Exception:
        return {"type": "FeatureCollection", "features": []}


def create_line_feature(line_geom, level_value):
    """Create GeoJSON feature from LineString geometry."""
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


def create_byte_webp(u_data: np.ndarray, v_data: np.ndarray, output_size=(512, 512)) -> Image.Image:
    """
    Create byte WebP image from U/V wind components for WebGL particle systems.
    R channel = U component (scaled and offset)
    G channel = V component (scaled and offset) 
    B channel = Wind speed magnitude

    Args:
        u_data: 2D array of U wind component
        v_data: 2D array of V wind component
        output_size: Output image dimensions (width, height)

    Returns:
        PIL Image with encoded wind data
    """
    # Resize data to target dimensions
    original_shape = u_data.shape
    zoom_factors = (output_size[1] / original_shape[0],
                    output_size[0] / original_shape[1])

    u_resized = zoom(u_data, zoom_factors, order=1)
    v_resized = zoom(v_data, zoom_factors, order=1)

    # Calculate wind speed magnitude
    speed = np.sqrt(u_resized**2 + v_resized**2)

    # Calculate dynamic range from actual data
    u_min, u_max = np.min(u_resized), np.max(u_resized)
    v_min, v_max = np.min(v_resized), np.max(v_resized)
    range_max = max(abs(u_min), abs(u_max), abs(v_min), abs(v_max))

    # Normalize U/V components to 0-255 range using dynamic range
    u_normalized = np.clip((u_resized + range_max) /
                           (2 * range_max) * 255, 0, 255).astype(np.uint8)
    v_normalized = np.clip((v_resized + range_max) /
                           (2 * range_max) * 255, 0, 255).astype(np.uint8)

    # Normalize speed to 0-255 using dynamic max speed
    speed_max = np.max(speed)
    speed_normalized = np.clip(
        speed / speed_max * 255, 0, 255).astype(np.uint8)

    # Stack into RGB array
    byte_array = np.stack(
        [u_normalized, v_normalized, speed_normalized], axis=-1)

    return Image.fromarray(byte_array, mode='RGB')


# =========================================
# ENDPOINT FACTORY FUNCTIONS
# =========================================

def create_webp_endpoint(param, palette):
    """
    Factory function to create WebP image endpoint.

    Args:
        file_path: GRIB file path(s)
        param: Weather parameter name(s)
        palette: Color palette for visualization

    Returns:
        Endpoint function that serves WebP images
    """
    def endpoint(time_param: str, cache: bool = True, width: int = None, height: int = None):
        """
        Get weather map as WebP image.

        Args:
            time_param: UTC time in format YYYYMMDDHH
            cache: If True, use cached image if available
            width: Output image width (optional, uses original if not specified)
            height: Output image height (optional, uses original if not specified)

        Returns:
            WebP image file
        """
        parse_time_param(time_param)

        # Build cache filename with dimensions
        size_suffix = f"_{width}x{height}" if width and height else ""

        # Handle vector data (wind with U and V components)
        if isinstance(param, list) and len(param) == 2:
            webp_path = os.path.join(
                "static", f"wind_{time_param}{size_suffix}_rgb.webp")

            if os.path.exists(webp_path) and cache:
                return FileResponse(webp_path, media_type="image/webp")

            # Download U and V components
            u_data = download_grib(param[0], cache, time_param)
            v_data = download_grib(param[1], cache, time_param)

            # Calculate wind speed magnitude
            u = np.nan_to_num(u_data.values)
            v = np.nan_to_num(v_data.values)
            wind_speed = np.sqrt(u**2 + v**2)

            # Apply color palette to wind speed
            rgb_array = apply_palette_to_data_vectorized(wind_speed, palette)
            img = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')

            # Resize if dimensions specified
            if width and height:
                img = img.resize((width, height), Image.LANCZOS)

            img.save(webp_path, format="WEBP", quality=95, method=6)
            return FileResponse(webp_path, media_type="image/webp")

        # Handle scalar data
        else:
            webp_path = os.path.join(
                "static", f"{param}_{time_param}{size_suffix}_rgb.webp")

            if os.path.exists(webp_path) and cache:
                return FileResponse(webp_path, media_type="image/webp")

            # Download data
            data = download_grib(param, cache, time_param)
            array = np.nan_to_num(data.values, nan=np.nanmean(data.values))

            # Se for precipitação, converte para mm e aplica log
            rgb_array = apply_palette_to_data_vectorized(array, palette)
            img = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')

            # Resize if dimensions specified
            if width and height:
                img = img.resize((width, height), Image.LANCZOS)

            img.save(webp_path, format="WEBP", quality=95, method=6)
            return FileResponse(webp_path, media_type="image/webp")

    return endpoint


def create_isolines_endpoint(param: str):
    """
    Factory function to create GeoJSON isolines endpoint.

    Args:
        file_path: GRIB file path
        param: Weather parameter name

    Returns:
        Endpoint function that serves GeoJSON isolines
    """
    def endpoint(time_param: str, cache: bool = True, background_tasks: BackgroundTasks = None, request: Request = None):
        """
        Get GeoJSON isolines for mean sea level pressure.

        Args:
            time_param: UTC time in format YYYYMMDDHH
            cache: If True, use cached GeoJSON if available

        Returns:
            GeoJSON FeatureCollection (preview in Swagger, full file otherwise)
        """
        parse_time_param(time_param)

        # Download data
        data = download_grib(param, cache, time_param)
        array = np.nan_to_num(data.values, nan=np.nanmean(data.values))

        # Calculate appropriate contour interval
        valid_values = array[~np.isnan(array)]
        data_min = float(np.min(valid_values))
        data_max = float(np.max(valid_values))
        suggested_interval = max(1, round((data_max - data_min) / 50, 1))

        geojson_path = os.path.join(
            "static", f"{param}_{time_param}_{suggested_interval}_isolines.json")

        # Schedule cache cleanup
        if background_tasks:
            background_tasks.add_task(clean_old_cache, param)

        # Check if request is from Swagger UI
        is_swagger = request and "/docs" in request.headers.get("referer", "")

        # Return cached file if available
        if os.path.exists(geojson_path) and cache:
            if is_swagger:
                # Return preview for Swagger (first 10 features)
                with open(geojson_path, 'r') as f:
                    full_data = json.load(f)
                return {
                    "type": "FeatureCollection",
                    "features": full_data["features"][:10],
                    "note": f"Swagger preview: 10 of {len(full_data['features'])} features"
                }
            return FileResponse(geojson_path, media_type="application/json")

        # Generate new isolines
        lat_coords = data.coords.get('latitude', data.coords.get('lat'))
        lon_coords = data.coords.get('longitude', data.coords.get('lon'))
        lat_values = lat_coords.values if lat_coords is not None else None
        lon_values = lon_coords.values if lon_coords is not None else None

        geojson_data = generate_isolines_geojson(
            array, lat_values, lon_values, suggested_interval
        )

        # Save to cache
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


def create_byte_endpoint(params: list):
    """
    Factory function to create byte array WebP endpoint for WebGL.
    Suporta vetorial (U/V) e escalar (1 parâmetro).
    """
    def endpoint(time_param: str, cache: bool = True, width: int = 360, height: int = 180):
        parse_time_param(time_param)

        # Vetorial (ex: wind)
        if isinstance(params, list) and len(params) == 2:
            byte_array_path = os.path.join(
                "static", f"{params[0]}_{params[1]}_byte_array_{time_param}_{width}x{height}.webp")
            if os.path.exists(byte_array_path) and cache:
                return FileResponse(byte_array_path, media_type="image/webp")
            u_data = download_grib(params[0], cache, time_param)
            v_data = download_grib(params[1], cache, time_param)
            u_array = np.nan_to_num(u_data.values, nan=0)
            v_array = np.nan_to_num(v_data.values, nan=0)
            byte_webp = create_byte_webp(
                u_array, v_array, output_size=(width, height))
            byte_webp.save(byte_array_path, format="WEBP", optimize=True)
            return FileResponse(byte_array_path, media_type="image/webp")
        # Escalar
        else:
            byte_array_path = os.path.join(
                "static", f"{params}_byte_array_{time_param}_{width}x{height}.webp")
            if os.path.exists(byte_array_path) and cache:
                return FileResponse(byte_array_path, media_type="image/webp")
            data = download_grib(params, cache, time_param)
            array = np.nan_to_num(data.values, nan=np.nanmean(data.values))
            # Se for precipitação, converte para mm e aplica log
            if params == "tp":
                array_mm = array * 1000.0
                arr_resized = zoom(
                    array_mm, (height / array_mm.shape[0], width / array_mm.shape[1]), order=1)
                arr_log = np.log1p(arr_resized)
                arr_min, arr_max = np.min(arr_log), np.max(arr_log)
                if arr_max - arr_min < 1e-6:
                    arr_norm = np.zeros_like(arr_log, dtype=np.uint8)
                else:
                    arr_norm = np.clip(
                        (arr_log - arr_min) / (arr_max - arr_min) * 255, 0, 255).astype(np.uint8)
            else:
                # Resize para output_size
                original_shape = array.shape
                zoom_factors = (
                    height / original_shape[0], width / original_shape[1])
                arr_resized = zoom(array, zoom_factors, order=1)
                # Normaliza para 0-255
                arr_min, arr_max = np.min(arr_resized), np.max(arr_resized)
                if arr_max - arr_min < 1e-6:
                    arr_norm = np.zeros_like(arr_resized, dtype=np.uint8)
                else:
                    arr_norm = np.clip(
                        (arr_resized - arr_min) / (arr_max - arr_min) * 255, 0, 255).astype(np.uint8)
            # Cria RGB: R=valor normalizado, G=0, B=0
            rgb_array = np.stack([arr_norm, np.zeros_like(
                arr_norm), np.zeros_like(arr_norm)], axis=-1)
            img = Image.fromarray(rgb_array, mode='RGB')
            img.save(byte_array_path, format="WEBP", optimize=True)
            return FileResponse(byte_array_path, media_type="image/webp")
    return endpoint


def create_info_endpoint(param: str, units: str):
    """
    Factory function to create data information endpoint.

    Args:
        file_path: GRIB file path(s)
        param: Weather parameter name(s)
        units: Measurement units

    Returns:
        Endpoint function that serves metadata and statistics
    """
    def endpoint(time_param: str, cache: bool = True):
        """
        Get metadata and statistics for weather parameter.
        Includes value range, mean, std deviation, spatial bounds, and resolution.

        Args:
            time_param: UTC time in format YYYYMMDDHH
            cache: If True, use cached GRIB data if available

        Returns:
            JSON object with data statistics and metadata
        """
        parse_time_param(time_param)

        # Handle vector data (wind with U and V components)
        if isinstance(param, list) and len(param) == 2:
            # Download U and V components
            u_data = download_grib(param[0], cache, time_param)
            v_data = download_grib(param[1], cache, time_param)

            u = np.nan_to_num(u_data.values)
            v = np.nan_to_num(v_data.values)
            values = np.sqrt(u**2 + v**2)  # Wind speed magnitude

            # Use U's coordinates for bounds
            lat_coords = u_data.coords.get(
                'latitude', u_data.coords.get('lat'))
            lon_coords = u_data.coords.get(
                'longitude', u_data.coords.get('lon'))

            # Calculate U/V component ranges
            uMin, uMax = float(np.min(u)), float(np.max(u))
            vMin, vMax = float(np.min(v)), float(np.max(v))

        # Handle scalar data
        else:
            data = download_grib(param, cache, time_param)
            values = np.nan_to_num(data.values, nan=np.nanmean(data.values))

            lat_coords = data.coords.get('latitude', data.coords.get('lat'))
            lon_coords = data.coords.get('longitude', data.coords.get('lon'))

            uMin = uMax = vMin = vMax = None

        # Calculate spatial bounds and resolution
        bounds = [-180, -90, 180, 90]
        resolution = {"lat": 0.25, "lon": 0.25}

        if lat_coords is not None and lon_coords is not None:
            lat_values, lon_values = lat_coords.values, lon_coords.values
            bounds = [
                float(lon_values.min()),
                float(lat_values.min()),
                float(lon_values.max()),
                float(lat_values.max())
            ]

            if len(lat_values) > 1:
                resolution["lat"] = float(abs(lat_values[1] - lat_values[0]))
            if len(lon_values) > 1:
                resolution["lon"] = float(abs(lon_values[1] - lon_values[0]))

        # Calculate data statistics
        valid_values = values[~np.isnan(values)]
        data_min = float(np.min(valid_values))
        data_max = float(np.max(valid_values))

        result = {
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
                "data_shape": list(values.shape),
                "bounds": bounds,
                "resolution": resolution
            },
        }

        # Add U/V component ranges for wind data
        if uMin is not None:
            result["uMin"] = uMin
            result["uMax"] = uMax
            result["vMin"] = vMin
            result["vMax"] = vMax

        return result

    return endpoint


# =========================================
# DYNAMIC ENDPOINT REGISTRATION
# =========================================

# Register endpoints for each weather theme from configuration
for theme, cfg in themes.items():
    # WebP weather map endpoint
    app.get(
        f"/{theme}/{{time_param}}/data.color.webp",
        tags=[theme.replace('_', ' ').title()]
    )(
        create_webp_endpoint(cfg["variable"], cfg["palette"])
    )

    # GeoJSON isolines endpoint (only for mean sea level pressure)
    if theme == "mean_sea_level_pressure":
        app.get(
            f"/{theme}/{{time_param}}/isolines.geojson",
            tags=[theme.replace('_', ' ').title()]
        )(
            create_isolines_endpoint(cfg["variable"])
        )

    # Byte array WebP endpoint for all themes
    app.get(
        f"/{theme}/{{time_param}}/data.byte.webp",
        tags=[theme.replace('_', ' ').title()]
    )(
        create_byte_endpoint(cfg["variable"])
    )

    # Data information endpoint
    app.get(
        f"/{theme}/{{time_param}}/info",
        tags=[theme.replace('_', ' ').title()]
    )(
        create_info_endpoint(cfg["variable"], cfg["units"])
    )


# =========================================
# V1 API ENDPOINTS
# =========================================

@app.get("/v1/", tags=["v1"])
def get_v1_info():
    """
    Get API v1 information and available endpoints.
    
    Returns:
        JSON object with API v1 metadata and available endpoints
    """
    return {
        "api": "Tempo API",
        "version": "1.0.0",
        "description": "Weather data visualization and analysis API using ECMWF OpenData",
        "endpoints": {
            "themes": "/v1/themes",
            "health": "/v1/health",
            "weather_maps": "/v1/{theme}/{timestamp}/data.color.webp",
            "isolines": "/v1/{theme}/{timestamp}/isolines.geojson",
            "byte_data": "/v1/{theme}/{timestamp}/data.byte.webp",
            "metadata": "/v1/{theme}/{timestamp}/info"
        },
        "data_source": "ECMWF OpenData",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/v1/themes", tags=["v1"])
def get_v1_themes():
    """
    Get available weather themes and their configurations.
    
    Returns:
        JSON object with all available weather themes and their parameters
    """
    return {
        "themes": themes,
        "count": len(themes),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/v1/health", tags=["v1"])
def get_v1_health():
    """
    Health check endpoint for API v1.
    
    Returns:
        JSON object with health status and system information
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_directory": "static/",
        "themes_loaded": len(themes),
        "grib_files": len(glob.glob("static/grib/*.grib2"))
    }


@app.get("/v1/{theme}/{time_param}/data.color.webp", tags=["v1"])
def get_v1_webp_map(theme: str, time_param: str, cache: bool = True, width: int = None, height: int = None):
    """
    Get weather map as WebP image (v1 endpoint).
    
    Args:
        theme: Weather theme name (e.g., 'temperature', 'wind')
        time_param: UTC time in format YYYYMMDDHH
        cache: If True, use cached image if available
        width: Output image width (optional)
        height: Output image height (optional)
        
    Returns:
        WebP image file with weather visualization
    """
    if theme not in themes:
        raise HTTPException(status_code=404, detail=f"Theme '{theme}' not found")
    
    endpoint_func = create_webp_endpoint(themes[theme]["variable"], themes[theme]["palette"])
    return endpoint_func(time_param, cache, width, height)


@app.get("/v1/{theme}/{time_param}/isolines.geojson", tags=["v1"])
def get_v1_isolines(theme: str, time_param: str, cache: bool = True, background_tasks: BackgroundTasks = None, request: Request = None):
    """
    Get GeoJSON isolines for weather data (v1 endpoint).
    
    Args:
        theme: Weather theme name (only 'mean_sea_level_pressure' supported for isolines)
        time_param: UTC time in format YYYYMMDDHH
        cache: If True, use cached GeoJSON if available
        
    Returns:
        GeoJSON FeatureCollection with isolines
    """
    if theme not in themes:
        raise HTTPException(status_code=404, detail=f"Theme '{theme}' not found")
    
    if theme != "mean_sea_level_pressure":
        raise HTTPException(status_code=400, detail="Isolines only available for mean_sea_level_pressure theme")
    
    endpoint_func = create_isolines_endpoint(themes[theme]["variable"])
    return endpoint_func(time_param, cache, background_tasks, request)


@app.get("/v1/{theme}/{time_param}/data.byte.webp", tags=["v1"])
def get_v1_byte_data(theme: str, time_param: str, cache: bool = True, width: int = 360, height: int = 180):
    """
    Get byte array WebP for WebGL applications (v1 endpoint).
    
    Args:
        theme: Weather theme name
        time_param: UTC time in format YYYYMMDDHH
        cache: If True, use cached data if available
        width: Output width (default: 360)
        height: Output height (default: 180)
        
    Returns:
        WebP image with encoded byte data for WebGL
    """
    if theme not in themes:
        raise HTTPException(status_code=404, detail=f"Theme '{theme}' not found")
    
    endpoint_func = create_byte_endpoint(themes[theme]["variable"])
    return endpoint_func(time_param, cache, width, height)


@app.get("/v1/{theme}/{time_param}/info", tags=["v1"])
def get_v1_metadata(theme: str, time_param: str, cache: bool = True):
    """
    Get metadata and statistics for weather parameter (v1 endpoint).
    
    Args:
        theme: Weather theme name
        time_param: UTC time in format YYYYMMDDHH
        cache: If True, use cached data if available
        
    Returns:
        JSON object with data statistics and metadata
    """
    if theme not in themes:
        raise HTTPException(status_code=404, detail=f"Theme '{theme}' not found")
    
    endpoint_func = create_info_endpoint(themes[theme]["variable"], themes[theme]["units"])
    return endpoint_func(time_param, cache)


# =========================================
# STATIC AND ROOT ENDPOINTS
# =========================================

@app.get("/themes.json", tags=["Static"], response_class=FileResponse)
def get_themes_json():
    """
    Get weather themes configuration file.
    Contains available weather map types, parameters, and color palettes.

    Returns:
        JSON configuration file
    """
    return FileResponse("themes.json", media_type="application/json")


@app.get("/", tags=["Static"])
def redirect_to_docs():
    """
    Redirect root URL to interactive API documentation (Swagger UI).

    Returns:
        Redirect response to /docs
    """
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Static"])
def health_check():
    """
    Health check endpoint for monitoring API status.
    Returns current timestamp, cache directory, and number of loaded themes.

    Returns:
        JSON object with health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_directory": "static/",
        "themes_loaded": len(themes)
    }