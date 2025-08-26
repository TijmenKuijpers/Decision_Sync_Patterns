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
queue_1 = production_line.add_var("q1")
resource = production_line.add_var("r1")
#finished = production_line.add_var("finished")

log_timer = production_line.add_var("log_timer")
production_line.add_event([log_timer], [log_timer], behavior=lambda log_timer: [SimToken(0, delay=0.1)], name="timer")
log_timer.put(0)

# Define the events that can change the state-space of the system
def job_arrival(arrival):

    job = arrival["job"]+1
    new_arrival = {"job": job}

    # Increased arrival delay to be closer to processing time
    return [SimToken(new_arrival, delay=4), SimToken(arrival, delay=np.random.exponential(scale=5.0))]

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

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    
    # Create progress reporter
    progress_reporter = ProgressReporter(simtime, print_interval=0.1)  # Print every 10%
    pattern_reporter = PatternReporter(production_line, ["value"], ["hold-batch"], simulation_name="batching")

    print(f"Starting simulation for {simtime} time units...")
    production_line.simulate(simtime, [progress_reporter, pattern_reporter])
    print("Simulation completed!")
    
    pattern_df = pattern_reporter.get_state_df()

    
    # Save logs to excel
    if log:
        pattern_reporter.to_excel()
        print("Saved to excel")


