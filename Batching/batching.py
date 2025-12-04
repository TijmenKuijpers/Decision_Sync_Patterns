import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import pattern_simulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from simpn.simulator import SimToken
from pattern_simulator import SimPattern, BehaviorEventLogReporter
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
    delaygen = np.random.exponential(scale=6.0)
    # Increased arrival delay to be closer to processing time
    return [SimToken(new_arrival, delay=delaygen), SimToken(arrival, delay=delaygen)]#np.random.exponential(scale=5.0))]

def transportation(q1, resource):

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
log = False

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    function_event_log_reporter = BehaviorEventLogReporter(production_line, "batching_function_log.csv", separator=";")
    production_line.simulate(simtime, [function_event_log_reporter])

    # Save logs to excel
    if log:
        function_event_log_reporter.save_report()




