
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
     ".out" \
     ".venv/*" \
     "venv/*" \
     "env/*" \
     ".env" \
     "docs/BIDS_ANTsPyMM_Seamless_Automation.pdf" \
     "build/*"\
     ".DS_Store" \
     "**/.DS_Store" \
     ".git/*"
