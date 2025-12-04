import random
import sys
from pathlib import Path

# Add parent directory to path to import pattern_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))
from simpn.simulator import SimToken
from pattern_simulator import SimPattern, BehaviorEventLogReporter
from simpn.visualisation import Visualisation

# Set up the production line model
production_line = SimPattern(binding_priority = SimPattern.PRIORITY_BINDING)

# Define the variables that make up the state-space of the system
arrival = production_line.add_var("arrival")
queue_1 = production_line.add_var("q1", priority=lambda token: -token.value["value"])
resource = production_line.add_var("r1")

# Define the events that can change the state-space of the system
def job_arrival(arrival):
    # Generate a new arrival with uniform value distribution
    random_value = random.randint(100, 1000)
    job = arrival["job"]+1
    new_arrival = {"job": job, "value": random_value}

    return [SimToken(new_arrival, delay=5.0), SimToken(arrival, delay=5.0)]

def job_handling_guard(q, r):
    a_queue = arrival.queue.marking[0].value
    q_queue = queue_1.queue.marking[0].value
    enabled_q_queue = [token for token in q_queue if token.time <= production_line.clock]

    value_a = a_queue[0].value["value"]
    
    value_q = max([token.value["value"] for token in q_queue])

    if len(enabled_q_queue)>0:
        enabled_value_q = max([token.value["value"] for token in enabled_q_queue])
    else:
        enabled_value_q = 0

    return value_a < 1.5*enabled_value_q and value_q==enabled_value_q

production_line.add_event([arrival], [arrival, queue_1], job_arrival, name="job_arrival")
production_line.add_event([queue_1, resource], [resource], 
                          behavior= lambda q, r: [SimToken(r, delay=7)],
                          guard= job_handling_guard,
                          name="job_handling")

# Describe the initial state of the system
arrival.put({"job": 1, "value": 100})
resource.put({"worker_id": 1})

# After simulation, print some statistics
visualize = False
simulate = True
log = False

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    function_event_log_reporter = BehaviorEventLogReporter(production_line, "priority_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])
    
    if log:
        function_event_log_reporter.save_report()



