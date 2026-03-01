# Sử dụng Python 3.9 phiên bản slim cho runtime nhẹ
FROM python:3.9-slim

# Ngăn Python ghi file .pyc và bật log unbuffered
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Tạo thư mục làm việc trong container
WORKDIR /app

# Copy requirements trước để tận dụng cache của Docker layer
COPY requirements.txt /app/

# Nâng cấp pip và cài đặt dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào container
COPY . /app/

# Mở port 8501 (cổng mặc định của Streamlit)
EXPOSE 8501

# Chạy ứng dụng web
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
