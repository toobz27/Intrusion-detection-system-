FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 flask==3.1.0 && \
    pip install --no-cache-dir --default-timeout=300 --retries 10 numpy==1.26.4 && \
    pip install --no-cache-dir --default-timeout=300 --retries 10 joblib==1.3.2 && \
    pip install --no-cache-dir --default-timeout=300 --retries 10 scikit-learn==1.3.2
EXPOSE 5000
CMD ["python", "app.py"]