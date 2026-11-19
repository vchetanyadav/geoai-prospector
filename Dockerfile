# GeoAI-Prospector: pipeline + Streamlit app in one image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app

# rasterio / pyogrio / netCDF4 wheels bundle GDAL and HDF5, but their GDAL still links the
# system libexpat (XML) at runtime; libgomp is for xgboost. zlib is already in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 curl && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1.5 GB smaller than the CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY config ./config
COPY app ./app

# Build the outputs at image build time so the container serves a map immediately
ARG RUN_PIPELINE=1
RUN if [ "$RUN_PIPELINE" = "1" ]; then python -m geoai_prospector.pipeline; fi

EXPOSE 8501
HEALTHCHECK CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
