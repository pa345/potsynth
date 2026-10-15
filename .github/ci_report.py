#!/usr/bin/env python3
#
# ci_report.py -- turn test and build failures into GitHub Actions error
# annotations, so that they appear on the run summary page
#
# Usage: python .github/ci_report.py DIR...
#   searches each DIR for ctest logs (LastTest*.log) and check.sh logs
#   (*.configure.log, *.build.log)

import pathlib
import re
import sys

PATTERN = re.compile(r"FAIL|[Ee]rror|Sanitizer|runtime error|Traceback|Assertion|Exception|Segmentation|[Ee]xit code")
MAXLINES = 150


def annotate(title, lines):
    msg = "\n".join(lines[:MAXLINES])
    msg = msg.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={title}::{msg}")


for root in sys.argv[1:]:
    for log in sorted(pathlib.Path(root).rglob("*.log")):
        name = log.name
        if not (name.startswith("LastTest") or name.endswith((".configure.log", ".build.log"))):
            continue
        lines = log.read_text(errors="replace").splitlines()
        hits = [l for l in lines if PATTERN.search(l) and not l.startswith("PASS")]
        if hits:
            annotate(f"{log.relative_to(root)}", hits)
