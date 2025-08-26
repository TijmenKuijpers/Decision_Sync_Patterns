from simpn.simulator import SimProblem, SimToken
from simpn.reporters import SimpleReporter
from simpn.reporters import Reporter
from simpn.visualisation import Visualisation
import random
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import pattern_reporter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pattern_reporter import PatternReporter, ProgressReporter

assembly_system = SimProblem()

# Define the variables that make up the state-space of the system
arrival_chip = assembly_system.add_var("chip supply")
arrival_phone_case = assembly_system.add_var("phone case supply")

stock_chip = assembly_system.add_var("stock chip")
stock_phone_case = assembly_system.add_var("stock phone case")

log_timer = assembly_system.add_var("log_timer")
assembly_system.add_event([log_timer], [log_timer], behavior=lambda log_timer: [SimToken(0, delay=0.5)], name="timer")
log_timer.put(0)

# Define the events that cause arrivals
def chip_arrival(arrival):

    delay = np.random.exponential(scale=3.0)
    chip_id = arrival["chip_id"]+1
    new_arrival = {"chip_id": chip_id}

    return [SimToken(new_arrival, delay=2), SimToken(new_arrival, delay=delay)]

def phone_case_arrival(arrival):

    delay = np.random.exponential(scale=5.0)
    phone_case_id = arrival["phone_case_id"]+1
    new_arrival = {"phone_case_id": phone_case_id}

    return [SimToken(new_arrival, delay=5), SimToken(new_arrival, delay=delay)]

assembly_system.add_event([arrival_chip], [arrival_chip, stock_chip], 
                          behavior=chip_arrival, 
                          name="chip_arrival")

assembly_system.add_event([arrival_phone_case], [arrival_phone_case, stock_phone_case], 
                          behavior=phone_case_arrival, 
                          name="phone_case_arrival")

# Define the events that cause processing
def game_processing_guard(stock_chip):

    all_tokens =  stock_phone_case.queue.marking[0].value
    time_enabled = [token for token in all_tokens if token.time <= assembly_system.clock]
    waiting_tokens = [token for token in all_tokens if token.time > assembly_system.clock]

    if len(waiting_tokens) > 0:
        time_untill_new = waiting_tokens[0].time - assembly_system.clock
    else:
        time_untill_new = 100000
    
    return len(time_enabled) == 0 and time_untill_new > 2 
 
assembly_system.add_event([stock_chip], [], 
                          behavior= lambda stock_chip : [],
                          guard= game_processing_guard,
                          name="game_production")

assembly_system.add_event([stock_chip, stock_phone_case], [], 
                          behavior= lambda stock_chip, stock_phone_case: [],
                          name="phone_production")

# Describe the initial state of the system
arrival_chip.put({"chip_id": 0})
arrival_phone_case.put({"phone_case_id": 0})


visualize = False
simulate = True
log = True

if visualize:
    v = Visualisation(assembly_system)
    v.show()

if simulate:
    simtime = 10000
    
    pattern_reporter = PatternReporter(assembly_system, ["value"], ["choice"], simulation_name="choice")
    progress_reporter = ProgressReporter(simtime, print_interval=0.1)
    assembly_system.simulate(simtime, [pattern_reporter, progress_reporter])

    pattern_df = pattern_reporter.get_state_df()
    # State reporter
    print("\nProcess States:")
    print(pattern_df.head())

    if log:
        pattern_reporter.to_excel()
        print("Saved to excel")