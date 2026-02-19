"""portfoliotools Flask application — blueprint registration.

This module creates the Flask app and registers all service blueprints.
Served by gunicorn on port 10000 (Render Starter plan).
"""

from __future__ import annotations

import logging
import os

from flask import Flask

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # Register blueprints
    # ------------------------------------------------------------------
    from para_score_blueprint import para_score_bp

    app.register_blueprint(para_score_bp)

    logger.info("Registered blueprint: para_score_bp")

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
