import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import pattern_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from simpn.simulator import SimToken
from pattern_simulator import SimPattern, BehaviorEventLogReporter, GuardedAlignment
from simpn.visualisation import Visualisation

# Set up the production line model
production_line = SimPattern()

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
queue_1 = production_line.add_var("q1")
resource = production_line.add_var("r1")

# Define the events that can change the state-space of the system
def job_arrival(arrival):

    job = arrival["job"]+1
    new_arrival = {"job": job}

    # Increased arrival delay to be closer to processing time
    return [SimToken(new_arrival, delay=8), SimToken(arrival, delay=8)]#np.random.exponential(scale=5.0))]

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


production_line.add_event([arrival], [arrival, queue_1], job_arrival, name="job_arrival")
production_line.add_event([queue_1, resource], [resource], 
                          behavior= transportation,
                          guard= None,
                          name="transportation")

# Describe the initial state of the system
arrival.put({"job": 1})
resource.put({"truck_id": 1, "loading": False})

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
    function_event_log_reporter = BehaviorEventLogReporter(production_line, "batching_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])
    # Save logs to excel
    if log:
        function_event_log_reporter.save_report()
    
# analyse guarded alignment
if analysis:

        def time_until_next_enabled(queue):
            if len([token for token in queue if token.time > production_line.clock]) > 0:
                return min([token.time - production_line.clock for token in queue if token.time > production_line.clock])
            else:
                return 100
        
        def nr_tokens_enabled(queue):
            return len([token for token in queue if token.time <= production_line.clock])

        alignment = GuardedAlignment(production_line, "batching_function_log.csv", separator=";")
        result = alignment.alignment(functions=[lambda q1_queue_tokens: time_until_next_enabled(q1_queue_tokens), 
                                     lambda q1_queue_tokens: nr_tokens_enabled(q1_queue_tokens)])
        
        functions = ["time until next enabled", "q1 enabled nr tokens"]
        alignment.save_alignment(result, "batching_structural_alignment_result", functions)

