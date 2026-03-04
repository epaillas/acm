#!/bin/bash

cd $ACM_root

# add -r y to add various information, statistics, ...
pylint acm --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/pylint_report.txt
status=$?
cat quality/pylint_report.txt
#exit $status
echo "Result in acm/quality/pylint_report.txt"
