import subprocess
from pathlib import Path
from datetime import datetime

project = "Magma"
version = "1.0.0"
release = version
copyright = f"2024–{datetime.now().year}, Magma Team"

extensions = [
    "breathe",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
]

breathe_projects = {"Magma": str(Path(__file__).resolve().parent / "_doxygen" / "xml")}
breathe_default_project = "Magma"

exclude_patterns = ["_doxygen"]
suppress_warnings = ["misc.highlighting_failure"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
