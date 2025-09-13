
# Tempo API

This service provides meteorological data processing and caching for maritime and weather applications. It is built with FastAPI and is Docker-ready for easy deployment.

## Main Features
- **GRIB/NetCDF Data Processing**: Supports common meteorological file formats.
- **Dynamic API Endpoints**: Automatically exposes endpoints for each weather indicator defined in `themes.json`.
- **Advanced Libraries**: Uses GDAL, eccodes, and other scientific libraries for spatial and meteorological data manipulation.
- **Local Storage**: All processed data is stored in `/app/data`.
- **Swagger Documentation**: Interactive API docs available at `/docs`.

## How to Use

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the service:
    ```bash
    python main.py
    ```
3. Or use Docker:
    ```bash
    docker-compose up --build
    ```
4. Access the API documentation at [http://localhost:3000/docs](http://localhost:3000/docs)

## Endpoints Overview

### `/[theme]/{time_param}/data.webp`
Returns a colorized weather map as a WebP image for the given theme and timestamp.
- **Method:** GET
- **Params:**
   - `theme`: Indicator name (e.g., `temperature`, `mean_sea_level_pressure`)
   - `time_param`: Timestamp in `YYYYMMDDHH` format
- **Response:** WebP image
- **Example:** `/temperature/2025091312/data.webp`

### `/[theme]/{time_param}/isolines.geojson`
Returns GeoJSON isolines (contour lines) for the selected indicator and timestamp.
- **Method:** GET
- **Params:**
   - `theme`: Indicator name
   - `time_param`: Timestamp
- **Response:** GeoJSON FeatureCollection
- **Example:** `/mean_sea_level_pressure/2025091312/isolines.geojson`

### `/[theme]/{time_param}/info`
Returns metadata and statistics about the indicator for the given timestamp.
- **Method:** GET
- **Params:**
   - `theme`: Indicator name
   - `time_param`: Timestamp
- **Response:** JSON with parameter name, units, statistics (min, max, mean, std), spatial info
- **Example:** `/temperature/2025091312/info`

### `/themes.json`
Returns the configuration of available weather themes/indicators.
- **Method:** GET
- **Response:** JSON

### `/health`
Health check endpoint for monitoring the API status.
- **Method:** GET
- **Response:** JSON with status, timestamp, cache directory, and number of loaded themes

### `/`
Redirects to the Swagger documentation (`/docs`).

## Example Themes
The available indicators are defined in `themes.json`. Example:
```json
{
   "temperature": {
      "file": "temp_2m.grib2",
      "variable": "2t",
      "units": "K"
   },
   "mean_sea_level_pressure": {
      "file": "temp_msl.grib2",
      "variable": "msl",
      "units": "Pa"
   }
}
```

## Directory Structure
- `main.py`: Main API script
- `generate_cache.py`: Cache generation script
- `start.sh`: Docker startup script
- `themes.json`: Weather indicator configuration
- `static/`: Static files and generated outputs
- `data/`: Processed data storage

---
For questions or suggestions, contact the project maintainer.
