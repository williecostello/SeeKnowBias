from python:3.6-slim

WORKDIR /root

COPY requirements.txt /root

RUN pip install --upgrade pip setuptools && \
	pip install -r /root/requirements.txt --no-cache-dir

COPY . ./root/dash

EXPOSE 80 8051

CMD ["python", "./dl_model.py"]
CMD ["python", "./app.py"]