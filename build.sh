#!/bin/bash

python setup.py sdist bdist_wheel
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
pip install --upgrade -i https://test.pypi.org/simple/ bigDATA
