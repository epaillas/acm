#!/bin/bash

cd $ACM_root

#mypy acm > quality/mypy_report.txt
ty check acm > quality/type_report.txt
echo "Result in acm/quality/type_report.txt"