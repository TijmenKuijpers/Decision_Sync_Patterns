import random
import pandas as pd
from datetime import datetime
import numpy as np
import sys
import os

from simpn.simulator import SimProblem, SimToken
from simpn.reporters import FunctionEventLogReporter
from simpn.visualisation import Visualisation

# Add the Patterns directory to the path to import analysis_branch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_branch import GuardedAlignment

production_line = SimProblem(binding_priority = SimProblem.PRIORITY_BINDING)

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
q1 = production_line.add_var("q1")
r1 = production_line.add_var("r1")
r2 = production_line.add_var("r2")

# Define the events that can change the state-space of the system
def resource_delay(q, r):
    # Generate a delay with exponential distribution
    delay = round(np.random.exponential(scale=10), 2)

    return [SimToken(r, delay=delay)]

production_line.add_event([arrival, r1], [arrival, r1, q1], 
                          behavior = lambda a, r: [SimToken({"job": a["job"]+1}, delay=5.0), SimToken(r, delay=5.0), SimToken(a, delay=5.0)], 
                          guard = lambda a, r: len([token for token in q1.queue.marking[0].value]) < 5,
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
analysis = False

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000

    function_event_log_reporter = FunctionEventLogReporter(production_line, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Blocking/blocking_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])
    
    # Save logs to excel
    if log:
        #log_reporter.to_excel()
        function_event_log_reporter.save_report()
        print("Saved")
    
    if analysis:
        alignment = GuardedAlignment(production_line, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Blocking/blocking_function_log.csv", separator=";")
        result = alignment.alignment(functions=[lambda arrival_queue: len(arrival_queue), 
                                     lambda q1_queue: len(q1_queue)])
        functions = ["arrival queue length", "q1 queue length"]
        alignment.save_alignment(result, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Blocking/blocking_alignment_result", functions)

