import random
import sys
from pathlib import Path

# Add parent directory to path to import pattern_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from simpn.simulator import SimToken
from pattern_simulator import SimPattern, BehaviorEventLogReporter, GuardedAlignment
from simpn.visualisation import Visualisation

production_line = SimPattern(binding_priority = SimPattern.PRIORITY_BINDING)

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
queue_1 = production_line.add_var("q1")#, priority=lambda token: -token.value["value"])
resource = production_line.add_var("r1")

# Define the events that can change the state-space of the system
def job_arrival(arrival):
    # Generate a new arrival with uniform value distribution
    random_value = random.randint(100, 1000)
    job = arrival["job"]+1
    new_arrival = {"job": job, "value": random_value}

    return [SimToken(new_arrival, delay=5.0), SimToken(arrival, delay=5.0)]


production_line.add_event([arrival], [arrival, queue_1], job_arrival, name="job_arrival")
production_line.add_event([queue_1, resource], [resource], 
                          behavior= lambda q, r: [SimToken(r, delay=7)],
                          guard= None,
                          name="job_handling")

# Describe the initial state of the system
arrival.put({"job": 1, "value": 100})
resource.put({"worker_id": 1})

# After simulation, print some statistics
visualize = False
simulate = False
log = False
analysis = True

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    function_event_log_reporter = BehaviorEventLogReporter(production_line, "priority_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])

    if log:
        function_event_log_reporter.save_report()

if analysis:
    def is_max_enabled(queue):
        queue_values = [token.value["value"] for token in queue]
        queue_enabled_values = [token.value["value"] for token in queue if token.time <= production_line.clock]
        
        if len(queue_values) > 0:
            queue_max = max(queue_values)
        else:
            queue_max = -1 # Not the same as queue_enabled_max
        
        if len(queue_enabled_values) > 0:
            queue_enabled_max = max(queue_enabled_values)
        else:
            queue_enabled_max = 0
            
        return queue_max == queue_enabled_max
    
    alignment = GuardedAlignment(production_line, "priority_function_log.csv", separator=";")
    result = alignment.alignment(functions=[lambda arrival_queue: max(a["value"] for a in arrival_queue) if arrival_queue else 0, 
                                            lambda q1_queue: max(e["value"] for e in q1_queue) if q1_queue else 0,
                                            lambda q1_queue_tokens: is_max_enabled(q1_queue_tokens)])
    
    functions = ["max_arrival_value", "max_q1_value", "max_q1_value_is_enabled"]
    alignment.save_alignment(result, "priority_structural_alignment_result", functions)
