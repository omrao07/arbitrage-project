# backend/dashboard_api/app.py

from flask import Flask
from flask_cors import CORS

# Import Blueprints
from routes.weights import weights_bp
from routes.signals import signals_bp
from routes.pnl import pnl_bp
from routes.regions import regions_bp  # optional if using regional endpoints

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register routes
    app.register_blueprint(weights_bp, url_prefix="/api")
    app.register_blueprint(signals_bp, url_prefix="/api")
    app.register_blueprint(pnl_bp, url_prefix="/api")
    app.register_blueprint(regions_bp, url_prefix="/api")  # only if you added region router

    @app.route("/")
    def index():
        return {"message": "Macro Alpha Platform API is running"}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)