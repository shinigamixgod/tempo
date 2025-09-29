# Tempo API

Tempo API is a weather data visualization and analysis service. It provides weather maps, isolines, and statistics for various meteorological parameters such as temperature, wind, precipitation, and pressure. The API is built with FastAPI and uses ECMWF OpenData as its data source.

## Features

- Retrieves global forecast data from ECMWF OpenData
- Generates color weather maps and isolines (contour lines)
- Provides metadata and statistics for each weather parameter
- Offers easy-to-use REST API endpoints
- Includes interactive documentation at `/docs` (Swagger UI)
- Supports Docker and Python execution

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the API:**
   ```bash
   python main.py
   ```
3. **Or use Docker (recommended):**
   ```bash
   docker-compose up --build
   ```
4. **Open the documentation:**
   [http://localhost:3000/docs](http://localhost:3000/docs)

## Example: Vue + Deck.gl Integration

```vue
<script setup>
import { ref, onMounted } from "vue";
import axios from "axios";
const apiBaseUrl = "http://localhost:3000";
const weatherLayers = ref([]);
const loadWeatherThemes = async () => {
  const response = await axios.get(`${apiBaseUrl}/themes.json`);
  weatherLayers.value = Object.keys(response.data);
};
onMounted(loadWeatherThemes);
</script>
<template>
  <div>
    <div v-for="layer in weatherLayers" :key="layer">
      <code>{{ layer }}</code>
      <!-- Render your Deck.gl image or GeoJSON here -->
    </div>
  </div>
</template>
```

## Main Endpoints

- `/[layer]/{timestamp}/data.color.webp` — Weather map image (WebP, colorized)
- `/[layer]/{timestamp}/data.byte.webp` — Byte array image (WebP, for WebGL)
- `/[layer]/{timestamp}/isolines.geojson` — Isolines (GeoJSON, only for pressure)
- `/[layer]/{timestamp}/info` — Metadata and statistics
- `/themes.json` — List of available layers and palettes

Replace `{layer}` with one of: `temperature`, `mean_sea_level_pressure`, `total_precipitation`, `wind`  
Replace `{timestamp}` with a string like `2025091312` (format: YYYYMMDDHH)

## How to Add Layers to Your Map

### Temperature

![Temperature Map](/demo/temperature.png)

```js
import { BitmapLayer } from "@deck.gl/layers";
const imageUrl = `${apiBaseUrl}/temperature/{timestamp}/data.color.webp`;
const bounds = [-180, -90, 180, 90];
const temperatureLayer = new BitmapLayer({
  id: "temperature-layer",
  image: imageUrl,
  bounds,
  opacity: 0.6,
  pickable: true,
  visible: true,
  _imageCoordinateSystem: 1,
  autoHighlight: false,
});
deckOverlay.setProps({ layers: [temperatureLayer] });
```


### Mean Sea Level Pressure

![Mean Sea Level Pressure Map](/demo/mean_sea_level_pressure.png)

```js
import { BitmapLayer, GeoJsonLayer } from "@deck.gl/layers";
const imageUrl = `${apiBaseUrl}/mean_sea_level_pressure/{timestamp}/data.color.webp`;
const geojsonUrl = `${apiBaseUrl}/mean_sea_level_pressure/{timestamp}/isolines.geojson`;
const bounds = [-180, -90, 180, 90];
const mslLayer = new BitmapLayer({
  id: "msl-layer",
  image: imageUrl,
  bounds,
  opacity: 0.6,
  pickable: true,
  visible: true,
  _imageCoordinateSystem: 1,
  autoHighlight: false,
});
const isolinesLayer = new GeoJsonLayer({
  id: "msl-isolines",
  data: geojsonUrl,
  stroked: true,
  filled: false,
  getLineColor: [255, 255, 255],
  getLineWidth: 2,
});
deckOverlay.setProps({ layers: [mslLayer, isolinesLayer] });
```

### Total Precipitation

![Total Precipitation Map](/demo/total_precipitation.png)

```js
import { BitmapLayer } from "@deck.gl/layers";
const imageUrl = `${apiBaseUrl}/total_precipitation/{timestamp}/data.color.webp`;
const bounds = [-180, -90, 180, 90];
const precipLayer = new BitmapLayer({
  id: "precip-layer",
  image: imageUrl,
  bounds,
  opacity: 0.6,
  pickable: true,
  visible: true,
  _imageCoordinateSystem: 1,
  autoHighlight: false,
});
deckOverlay.setProps({ layers: [precipLayer] });
```

### Wind

![Wind Map](/demo/wind.png)

```js
import { BitmapLayer } from "@deck.gl/layers";
const imageUrl = `${apiBaseUrl}/wind/{timestamp}/data.color.webp`;
const bounds = [-180, -90, 180, 90];
const windLayer = new BitmapLayer({
  id: "wind-layer",
  image: imageUrl,
  bounds,
  opacity: 0.6,
  pickable: true,
  visible: true,
  _imageCoordinateSystem: 1,
  autoHighlight: false,
});
deckOverlay.setProps({ layers: [windLayer] });
```

## Always Use the `/info` Endpoint for Tooltips

Before rendering a layer or handling tooltips/clicks, always fetch `/[layer]/{timestamp}/info`. This endpoint provides the correct bounds, image size, and statistics for each layer and time. Use these values to map coordinates to pixels and retrieve the correct weather value. You can also display extra information from `/info`, such as min/max values, units, and timestamp, in your tooltip.

Example:

```js
// Fetch info before rendering or handling clicks
const infoData = await axios.get(`${apiBaseUrl}/temperature/{timestamp}/info`);
const bounds = infoData.data.spatial_info.bounds;
const width = infoData.data.spatial_info.data_shape[1];
const height = infoData.data.spatial_info.data_shape[0];

// Use these in your tooltip logic
const handleClick = async (info, layerId) => {
  if (!selectedLayer.value || !info?.picked || !info?.coordinate) return;
  const [lon, lat] = info.coordinate;
  const [minLon, minLat, maxLon, maxLat] = bounds;
  // Calculate pixel position
  const normalizedLon = (lon - minLon) / (maxLon - minLon);
  const normalizedLat = (maxLat - lat) / (maxLat - minLat);
  const px = Math.max(
    0,
    Math.min(width - 1, Math.floor(normalizedLon * width))
  );
  const py = Math.max(
    0,
    Math.min(height - 1, Math.floor(normalizedLat * height))
  );
  // Get value from image data (see your palette mapping)
  const realValue = await getRealValueAtPixel(layerId, px, py, paletteMap);
  if (realValue === null) return;
  const formatFn = config.formatTooltip || ((v) => v.toFixed(2));
  const displayValue = formatFn(realValue);
  // Show extra info from /info if you want
  showTooltip({
    value: displayValue,
    units: config.units,
    coordinate: [lon, lat],
    min: infoData.data.data_statistics.min_value,
    max: infoData.data.data_statistics.max_value,
    timestamp: infoData.data.timestamp,
  });
};
```

This ensures your tooltips are always accurate and provide useful information to users.

---

Questions or suggestions? Open an issue or contact the maintainer.

---


