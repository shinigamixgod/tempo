from datetime import datetime
from main import themes, create_webp_endpoint, create_isolines_endpoint

now = datetime.utcnow()
time_param = now.strftime("%Y%m%d%H")

for theme, cfg in themes.items():

    # WebP
    webp_endpoint = create_webp_endpoint(cfg["file"], cfg["variable"], cfg["palette"])
    webp_endpoint(time_param, force=True)

    # Isolines
    isolines_endpoint = create_isolines_endpoint(cfg["file"], cfg["variable"])
    isolines_endpoint(time_param, force=True)