#!/usr/bin/env python

with open('workshop.py', 'r') as f:
    for line in f:
        print(line, end='')
        if '%load' in line:
            filename = line.split('%load')[-1].strip()
            with open(filename, 'r') as f2:
                print(f2.read())
