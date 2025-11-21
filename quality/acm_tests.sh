#!/bin/bash

cd $ACM_root
coverage erase
rm -f quality/report_coverage*
rm -rf quality/html_cov*
coverage run --source=acm -m pytest tests/unit -v
coverage xml -i -o quality/report_coverage.xml
coverage html -i -d quality/html_coverage
coverage report
