FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Configuration de JupyterLab
RUN mkdir -p /root/.jupyter/lab/settings
RUN echo '{"@jupyterlab/notebook-extension:tracking": {"cellMetadata": {"tags": ["remove-cell"]}}}' > /root/.jupyter/lab/settings/overrides.json

# ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8888

CMD ["jupyter", "lab", "--notebook-dir=/app/notebooks", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]