#!/bin/bash

cd $ACM_root

ruff check acm --config quality/ruff.toml --output-format json > quality/ruff_report.json
ciqar -r ruff:quality/ruff_report.json  -s acm -o quality/ruff_report_html
echo "Result in acm/quality/ruff_report_html"
