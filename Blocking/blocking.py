import random
import pandas as pd
from datetime import datetime
import numpy as np
import sys
import os

from simpn.simulator import SimProblem, SimToken
from simpn.reporters import SimpleReporter
from simpn.reporters import Reporter
from simpn.visualisation import Visualisation

# Add the parent directory to the path to import pattern_reporter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pattern_reporter import PatternReporter, ProgressReporter

production_line = SimProblem()

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
q1 = production_line.add_var("q1")
r1 = production_line.add_var("r1")
r2 = production_line.add_var("r2")

# Define the events that can change the state-space of the system
def resource_delay(q, r):
    # Generate a delay with exponential distribution
    delay = np.random.exponential(scale=10)

    return [SimToken(r, delay=delay)]

production_line.add_event([arrival, r1, q1.queue], [arrival, r1, q1, q1.queue], 
                          behavior = lambda a, r, q: [SimToken({"job": a["job"]+1}, delay=5.0), SimToken(r, delay=5.0), SimToken(a, delay=5.0), q], 
                          guard = lambda a, r, q: len(q) < 5,
                          name="pre_processing")

production_line.add_event([q1, r2], [r2], 
                          behavior= resource_delay,
                          name="processing")

# Describe the initial state of the system
arrival.put({"job": 1})
r1.put({"machine_id": 1})
r2.put({"machine_id": 2})


# After simulation, print some statistics
visualize = False
simulate = True
log = True

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    pattern_reporter = PatternReporter(production_line, ["value"], ["blocking"], simulation_name="blocking")
    progress_reporter = ProgressReporter(simtime, print_interval=0.1)
    production_line.simulate(simtime, [pattern_reporter, progress_reporter])
    
    pattern_df = pattern_reporter.get_state_df()

    # State reporter
    print("\nProcess States:")
    print(pattern_df.head())

    # Save logs to excel
    if log:
        #log_reporter.to_excel()
        pattern_reporter.to_excel()
        print("Saved to excel")


