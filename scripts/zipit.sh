
zip -r ~/Downloads/antsxmmclean.zip . \
  -x "antsxmm.egg-info/*" \
     "**/antsxmm.egg-info/*" \
     "**/node_modules/*" "**/.git/*" "**/__pycache__/*" \
     "__pycache__/*" \
     "**/__pycache__/*" \
     "*.pyc" \
     "*.pyo" \
     ".pytest_cache/*" \
     "**/.pytest_cache/*" \
     ".coverage" \
     ".coverage.*" \
     "htmlcov/*" \
     ".mypy_cache/*" \
     "**/.mypy_cache/*" \
     ".ruff_cache/*" \
     "**/.ruff_cache/*" \
     ".tox/*" \
     ".venv/*" \
     "venv/*" \
     "env/*" \
     ".env" \
     ".DS_Store" \
     "**/.DS_Store" \
     ".git/*"
