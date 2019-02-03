#!/bin/sh
pip freeze | grep -Eo '(.*==)' | sed 's/==//g' >requirements.txt
