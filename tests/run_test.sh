#!/bin/bash

# generate report
pytest --cov-report term --cov=src tests/ --cov-report xml:cov.xml

# upload report to codacy
bash <(curl -Ls https://coverage.codacy.com/get.sh) report -r cov.xml