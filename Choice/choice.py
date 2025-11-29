import random
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

from simpn.simulator import SimProblem, SimToken
from simpn.reporters import  FunctionEventLogReporter
from simpn.visualisation import Visualisation


# Add the parent directory to the path to import pattern_reporter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_branch import GuardedAlignment

assembly_system = SimProblem()

# Define the variables that make up the state-space of the system
arrival_chip = assembly_system.add_var("chip supply")
arrival_phone_case = assembly_system.add_var("phone case supply")

stock_chip = assembly_system.add_var("stock chip")
stock_phone_case = assembly_system.add_var("stock_phone_case")

phone_resource = assembly_system.add_var("phone resource")
game_resource = assembly_system.add_var("game resource")

log_timer = assembly_system.add_var("log_timer")
assembly_system.add_event([log_timer], [log_timer], behavior=lambda log_timer: [SimToken(0, delay=0.5)], name="timer")
log_timer.put(0)

# Define the events that cause arrivals
def chip_arrival(arrival):

    delay = np.random.exponential(scale=3.0)
    chip_id = arrival["chip_id"]+1
    new_arrival = {"chip_id": chip_id}

    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]

def phone_case_arrival(arrival):

    delay = np.random.exponential(scale=5.0)
    phone_case_id = arrival["phone_case_id"]+1
    new_arrival = {"phone_case_id": phone_case_id}

    return [SimToken(new_arrival, delay=delay), SimToken(new_arrival, delay=delay)]

assembly_system.add_event([arrival_chip], [arrival_chip, stock_chip], 
                          behavior=chip_arrival, 
                          name="chip_arrival")

assembly_system.add_event([arrival_phone_case], [arrival_phone_case, stock_phone_case], 
                          behavior=phone_case_arrival, 
                          name="phone_case_arrival")

# Define the events that cause processing
def game_processing_guard(stock_chip, game_resource):

    all_tokens =  stock_phone_case.queue.marking[0].value
    time_enabled = [token for token in all_tokens if token.time <= assembly_system.clock]
    waiting_tokens = [token for token in all_tokens if token.time > assembly_system.clock]

    if len(waiting_tokens) > 0:
        time_untill_new = waiting_tokens[0].time - assembly_system.clock
    else:
        time_untill_new = 100000
    
    return time_untill_new > 2 #and len(time_enabled) == 0
 
assembly_system.add_event([stock_chip, game_resource], [game_resource], 
                          behavior= lambda stock_chip, game_resource: [SimToken(game_resource, delay=1)],
                          guard= game_processing_guard,
                          name="game_production")

assembly_system.add_event([stock_chip, stock_phone_case, phone_resource], [phone_resource], 
                          behavior= lambda stock_chip, stock_phone_case, phone_resource: [SimToken(phone_resource, delay=1)],
                          name="phone_production")

# Describe the initial state of the system
arrival_chip.put({"chip_id": 0})
arrival_phone_case.put({"phone_case_id": 0})
game_resource.put({"game_id": 1})
phone_resource.put({"phone_id": 1})

# After simulation, print some statistics
visualize = False
simulate = True
log = True
analysis = False

if visualize:
    v = Visualisation(assembly_system)
    v.show()

if simulate:
    simtime = 10000
    function_event_log_reporter = FunctionEventLogReporter(assembly_system, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Choice/choice_function_log.csv", separator=";")
    
    if not analysis:
        assembly_system.simulate(simtime, [function_event_log_reporter])

    # Save logs to excel
    if log:
        function_event_log_reporter.save_report()
        print("Saved")
    
    # analyse guarded alignment
    if analysis:
        alignment = GuardedAlignment(assembly_system, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Choice/choice_function_log.csv", separator=";")
        result = alignment.alignment(functions=[lambda arrival_queue: len(arrival_queue), 
                                     lambda q1_queue: len(q1_queue)])
        
        functions = ["arrival queue length", "q1 queue length"]
        alignment.save_alignment(result, "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/Choice/choice_structural_alignment_result", functions)