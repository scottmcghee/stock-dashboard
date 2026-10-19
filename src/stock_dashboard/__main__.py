"""``python -m stock_dashboard [streamlit options]``: serve the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
