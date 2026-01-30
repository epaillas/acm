#!/bin/bash

cd $ACM_root

mypy acm > quality/mypy_report.txt
echo "Result in acm/quality/mypy_report.txt"