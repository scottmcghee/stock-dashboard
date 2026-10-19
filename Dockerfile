# syntax=docker/dockerfile:1
# Base images are pinned by digest for reproducible builds; Dependabot bumps them.
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170 AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170
RUN useradd --create-home --uid 1000 dashboard
# Only the virtualenv is copied, owned by root: the runtime user can't modify the code.
COPY --from=build /app/.venv /app/.venv
# Headless, no usage statistics, the viewer toolbar (no developer "Deploy"
# menu), and a fixed browser address: without one, a headless Streamlit looks up
# the machine's public IP to print an "External URL".
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer STREAMLIT_BROWSER_SERVER_ADDRESS=localhost
USER dashboard
WORKDIR /home/dashboard
EXPOSE 8501
CMD ["stock-dashboard", "--server.address=0.0.0.0"]
