FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/uploads data/vectorstore
EXPOSE 8501 8000
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0"]
