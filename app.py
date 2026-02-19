"""portfoliotools Flask application — blueprint registration.

This module creates the Flask app and registers all service blueprints.
Served by gunicorn on port 10000 (Render Starter plan).
"""

from flask import Flask

from para_score_blueprint import para_score_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(para_score_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
