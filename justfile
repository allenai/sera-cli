# Format code
format:
    ruff format .
    ruff check --select I --fix .

# Lint code
lint:
    ruff check .
    ty check .

# Run required precommit checks
precommit: format lint

# Build the package
build:
    uv build

# Build for publishing (disables uv.sources)
build-publish:
    rm -r dist/*
    uv build --no-sources

# Publish to PyPI. Requires setting `UV_PUBLISH_TOKEN` env var with your PyPI token
publish: build-publish
    uv publish

# Dry run - just build without publishing
dry-run: build-publish
    @echo "Build artifacts in dist/"
