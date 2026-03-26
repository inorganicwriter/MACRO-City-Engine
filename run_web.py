from __future__ import annotations

"""Start dashboard web server."""

from web.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
