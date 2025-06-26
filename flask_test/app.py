

from datetime import timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)


@app.route('/')
def index():
    """Page principale avec le formulaire."""
    return "hello"
if __name__ == '__main__':
    print("🚀 Serveur Flask démarré en mode dev")
    print("🌐 Accédez à: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)