################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

from experiment import Experiment
import sys
import warnings
warnings.filterwarnings("ignore")
# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py task-1-default-config`
if __name__ == "__main__":
    exp_name = 'task-1-default-config'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()
