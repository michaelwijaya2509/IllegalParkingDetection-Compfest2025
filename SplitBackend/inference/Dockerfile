# --- Base ringan + ffmpeg untuk OpenCV VideoCapture (RTSP/HLS) ---
FROM python:3.10-slim

# Lib runtime yang umum dibutuhkan OpenCV/ffmpeg/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install deps dulu biar layer cache efektif ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy source + model + zona ---
# Pastikan file model YOLO & model perilaku (.pt/.pkl) dan folder config/ ikut dicopy.
COPY . .

# --- ENV runtime untuk stabilitas inference di container ---
# Hindari limit thread OpenMP yang berlebihan (CPU kecil di Cloud Run)
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # pastikan stream tidak kena buffer stdout
    PYTHONUNBUFFERED=1 \
    # mencegah error "KMP_DUPLICATE_LIB_OK" (karena tidak lewat _main_)
    KMP_DUPLICATE_LIB_OK=TRUE \
    # cache ultralytics di path yang bisa ditulis
    ULTRALYTICS_SETTINGS=/app/.ultralytics \
    # port default Cloud Run
    PORT=8080

# --- Jalankan via gunicorn (perlu worker-class thread utk stream MJPEG/SSE) ---
# sync worker default cenderung nge-blok; gthread lebih aman untuk stream generator.
# timeout 0 = jangan kill koneksi panjang (MJPEG/SSE)
CMD exec gunicorn app:app \
  --bind :$PORT \
  --workers 1 \
  --threads 16 \
  --worker-class gthread \