#!/bin/bash

cd $ACM_root
coverage erase
# xml is for SonarQub
#rm -f quality/coverage_report.xml
rm -rf quality/coverage_report_html
coverage run --source=acm -m pytest tests/acm -vv
#coverage xml -i -o quality/coverage_report.xml
coverage html -i -d quality/coverage_report_html
coverage report
echo "Result in acm/quality/coverage_report_html"