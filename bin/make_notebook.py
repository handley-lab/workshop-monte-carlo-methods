#!/bin/bash

py2nb workshop.py
jupyter nbconvert --execute --to notebook --inplace workshop.ipynb



