#!/bin/bash

cd $ACM_root/tests
coverage erase
# xml is for SonarQub
#rm -f quality/coverage_report.xml
pwd 
rm -rf ../quality/coverage_report_html
coverage run --source=../acm -m pytest acm -vv -s
#coverage xml -i -o quality/coverage_report.xml
coverage html -i -d ../quality/coverage_report_html
coverage report
echo "Result in acm/quality/coverage_report_html"
