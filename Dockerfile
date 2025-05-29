# 1. Use a lightweight Python base
FROM python:3.9-slim

# Install OS dependencies for LightGBM (and others you might need)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set a working directory
WORKDIR /app

# 3. Copy and install only dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code
COPY . .
RUN chmod +x ./start.sh

# 5. Expose the port your app listens on
EXPOSE 8080

ENTRYPOINT ["./start.sh"]

