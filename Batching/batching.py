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

production_line = SimProblem()

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
queue_1 = production_line.add_var("q1")
resource = production_line.add_var("r1")

#log_timer = production_line.add_var("log_timer")
#production_line.add_event([log_timer], [log_timer], behavior=lambda log_timer: [SimToken(0, delay=0.1)], name="timer")
#log_timer.put(0)

# Define the events that can change the state-space of the system
def job_arrival(arrival):

    job = arrival["job"]+1
    new_arrival = {"job": job}
    delaygen = np.random.exponential(scale=6.0)
    # Increased arrival delay to be closer to processing time
    return [SimToken(new_arrival, delay=delaygen), SimToken(arrival, delay=delaygen)]#np.random.exponential(scale=5.0))]

def transportation(q1, resource):

    #print(queue_1.queue.marking[0].value)
    all_tokens = queue_1.queue.marking[0].value
    time_enabled = [token for token in all_tokens if token.time <= production_line.clock]
    waiting_tokens = [token for token in all_tokens if token.time > production_line.clock]

    if len(time_enabled) > 0:
        delay_truck = 0
        resource['loading'] = True

    elif len(time_enabled) == 0: 
        delay_truck = 20
        resource["loading"] = False
    

    return [SimToken(resource, delay=delay_truck)]

def transportation_guard(q, r):

    all_tokens = queue_1.queue.marking[0].value
    time_enabled = [token for token in all_tokens if token.time <= production_line.clock]
    waiting_tokens = [token for token in all_tokens if token.time > production_line.clock]

    if len(waiting_tokens) > 0:
        time_untill_new = waiting_tokens[0].time - production_line.clock
    else:
        time_untill_new = 100

    return (len(time_enabled) > 3 or r["loading"] == True) and time_untill_new > 2

production_line.add_event([arrival], [arrival, queue_1], job_arrival, name="job_arrival")
production_line.add_event([queue_1, resource], [resource], 
                          behavior= transportation,
                          guard= transportation_guard,
                          name="transportation")

# Describe the initial state of the system
arrival.put({"job": 1})
resource.put({"truck_id": 1, "loading": False})

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
    function_event_log_reporter = FunctionEventLogReporter(production_line, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Batching/batching_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])

    # Save logs to excel
    if log:
        function_event_log_reporter.save_report()
        print("Saved")

    if analysis:
        alignment = GuardedAlignment(production_line, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Batching/batching_function_log.csv", separator=";")
        result = alignment.alignment(functions=[lambda arrival_queue: len(arrival_queue), 
                                     lambda q1_queue: len(q1_queue)])
        functions = ["arrival queue length", "q1 queue length"]
        alignment.save_alignment(result, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Batching/batching_alignment_result", functions)


