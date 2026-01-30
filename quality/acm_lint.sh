#!/bin/bash

cd $ACM_root
pwd
# add -r y to add various information, statistics, ...
#pylint acm --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/report_pylint.txt
#status=$?
#cat quality/report_pylint.txt
#exit $status

ruff check acm --output-format json > quality/ruff_report.json
ciqar -r ruff:quality/ruff_report.json  -s acm -o quality/ruff_report_html
echo "Result in acm/quality/ruff_report_html"
