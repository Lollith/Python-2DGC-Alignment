from flask import Flask, request, render_template, redirect
import os
import tempfile
import numpy as np
import netCDF4 as nc
from get_data import read_cdf_to_npy

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    cdf_file = request.files['cdf_file']
    output_name = request.form['output_name']

    with tempfile.TemporaryDirectory() as tmp_dir:
        cdf_path = os.path.join(tmp_dir, cdf_file.filename)
        cdf_file.save(cdf_path)
        
        output_path = f"{output_name}.npy"
        read_cdf_to_npy(cdf_path, output_path)

    return f"Fichier converti : {output_path}"

if __name__ == '__main__':
    app.run(debug=True)