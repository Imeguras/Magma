#!/usr/bin/env python3
"""License compliance check: REUSE lint + GPLv3 text present + copyright years current.

Usage: python3 scripts/check-license.py   (requires `reuse` on PATH and full git history)
"""
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# LGPLv3 incorporates GPLv3 by reference, so its text must ship even though no file is tagged GPL.
REQUIRED_TEXTS = ["LICENSES/LGPL-3.0-or-later.txt", "LICENSES/GPL-3.0-or-later.txt"]
ALLOWED_UNUSED = {"GPL-3.0-or-later"}
SOURCE_GLOBS = ["*.c", "*.cpp", "*.h", "*.hpp", "*.hip", "*.proto"]
HEADER_RE = re.compile(r"SPDX-FileCopyrightText:\s*(\d{4})(?:-(\d{4}))?\s+(.+)")
OWNER = "João Vieira"

errors = []

# 1. REUSE compliance (every file has copyright + license; all license texts present)
report = json.loads(subprocess.run(["reuse", "lint", "--json"], capture_output=True, text=True).stdout)
for kind, items in report["non_compliant"].items():
    if not items:
        continue
    if kind == "unused_licenses":
        items = [i for i in items if i not in ALLOWED_UNUSED]
        if not items:
            continue
    errors.append(f"reuse: {kind}: {items}")

# 2. Required license texts
for path in REQUIRED_TEXTS:
    if not Path(path).is_file():
        errors.append(f"missing license text: {path}")

# 3. Own source files carry a header whose year range reaches the file's last commit
# Files declared as third-party overrides in REUSE.toml keep their upstream notice.
with open("REUSE.toml", "rb") as fh:
    annotations = tomllib.load(fh).get("annotations", [])
third_party = [p for a in annotations if a.get("precedence") == "override" for p in ([a["path"]] if isinstance(a["path"], str) else a["path"])]
files = subprocess.check_output(["git", "ls-files", *SOURCE_GLOBS], text=True).split()
for f in files:
    if any(fnmatch.fnmatch(f, p) for p in third_party):
        continue
    head = Path(f).read_text(encoding="utf-8", errors="replace")[:1024]
    m = HEADER_RE.search(head)
    if not m:
        if "Copyright" not in head:  # third-party files keep their original notice (declared in REUSE.toml)
            errors.append(f"{f}: no SPDX-FileCopyrightText header")
        continue
    if OWNER not in m.group(3):
        continue  # third-party file, years maintained upstream
    end = int(m.group(2) or m.group(1))
    last = subprocess.check_output(["git", "log", "-1", "--format=%ad", "--date=format:%Y", "--", f], text=True).strip()
    if last and end < int(last):
        errors.append(f"{f}: copyright year {end} older than last change ({last})")

for e in errors:
    print(f"::error::{e}")
print(f"license check: {len(errors)} problem(s)")
sys.exit(1 if errors else 0)
