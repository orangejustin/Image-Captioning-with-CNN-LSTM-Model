#!/bin/bash

# call main.py with different config files and just run this script so that
# you don't have to call each experiment individually and keep logging back in

# to run this file, type
# $ chmod +x script.sh
# $ ./script.sh


python main.py task-1-default-config

python main.py task-1-dropout-config

python main.py task-1-emb500-config


# create more configs and stack them here
