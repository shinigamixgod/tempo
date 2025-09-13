FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libeccodes-dev libeccodes-tools libgeos-dev gdal-bin libgdal-dev \
    python3-gdal g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data && chmod 777 /app/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy startup script and make executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 3000

# Run startup script
CMD ["/app/start.sh"]
