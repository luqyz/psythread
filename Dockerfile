FROM python:3.11-slim

WORKDIR /app

# Without this, Python buffers stdout/stderr in a container (non-TTY)
# environment -- print() output can sit in memory indefinitely instead of
# reaching `gcloud run services logs read`, since gunicorn's worker process
# never exits to force a flush. This made debugging (translation errors,
# Firestore write status, etc.) unreliable until fixed.
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "cd app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120"]