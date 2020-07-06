FROM python:3.6

WORKDIR /

COPY requirements.txt .

RUN pip install --upgrade pip setuptools && \
	pip install -r /requirements.txt --no-cache-dir

COPY ./ ./

EXPOSE 8050


CMD ["python", "./app.py"]