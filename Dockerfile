FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /root/.jupyter
COPY config/jupyter_notebook_config.py /root/.jupyter/



EXPOSE 8000


CMD ["jupyter", "lab", "--notebook-dir=/app/notebooks", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]