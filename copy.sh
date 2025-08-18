#!/bin/bash

TIMETAG=$1

mkdir -p "$TIMETAG"

cp flight.ipynb "$TIMETAG/"
cp -r src/ "$TIMETAG/src"

git switch main