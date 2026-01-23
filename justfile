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
    uv build --no-sources

# Publish to PyPI
publish: build-publish
    uv publish

# Dry run - just build without publishing
dry-run: build-publish
    @echo "Build artifacts in dist/"
