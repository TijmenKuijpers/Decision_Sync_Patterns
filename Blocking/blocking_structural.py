import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import pattern_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from simpn.simulator import SimToken
from pattern_simulator import SimPattern, BehaviorEventLogReporter, GuardedAlignment
from simpn.visualisation import Visualisation

# Set up the production line model
production_line = SimPattern(binding_priority = SimPattern.PRIORITY_BINDING)

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
q1 = production_line.add_var("q1")
r1 = production_line.add_var("r1")
r2 = production_line.add_var("r2")

# Define the events that can change the state-space of the system
production_line.add_event([arrival, r1], [arrival, r1, q1], 
                          behavior = lambda a, r: [SimToken({"job": a["job"]+1}, delay=5.0), SimToken(r, delay=5.0), SimToken(a, delay=5.0)], 
                          guard = None,#lambda a, r: len([token for token in q1.queue.marking[0].value]) < 5,
                          name="pre_processing")

production_line.add_event([q1, r2], [r2], 
                          behavior= lambda q, r: [SimToken(r, delay=round(np.random.exponential(scale=10), 2))],
                          name="processing")

# Describe the initial state of the system
arrival.put({"job": 1})
r1.put({"machine_id": 1})
r2.put({"machine_id": 2})

# After simulation, print some statistics
visualize = False
simulate = False
log = False
analysis = True

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 100
    function_event_log_reporter = BehaviorEventLogReporter(production_line, "blocking_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])
    # Save logs to excel
    if log:
        function_event_log_reporter.save_report()
    
# analyse guarded alignment
if analysis:
    alignment = GuardedAlignment(production_line, "blocking_function_log.csv", separator=";")
    result = alignment.alignment(functions=[lambda arrival_queue: len(arrival_queue), 
                                    lambda q1_queue: len(q1_queue)])
    
    functions = ["arrival queue length", "q1 queue length"]
    alignment.save_alignment(result, "blocking_structural_alignment_result", functions)

