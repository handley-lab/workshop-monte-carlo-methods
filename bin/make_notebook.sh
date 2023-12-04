#!/bin/bash

py2nb workshop.py

./bin/fill_in_answers.py > workshop_answers.py
py2nb workshop_answers.py
jupyter nbconvert --execute --to notebook --inplace workshop_answers.ipynb
