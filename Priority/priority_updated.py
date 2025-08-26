import random
import pandas as pd
from datetime import datetime
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
queue_1 = production_line.add_var("q1", priority=lambda token: -token.value["value"])
resource = production_line.add_var("r1")
#finished = production_line.add_var("finished")

# Define the events that can change the state-space of the system
def job_arrival(arrival):
    # Generate a new arrival with uniform value distribution
    random_value = random.randint(100, 1000)
    job = arrival["job"]+1
    new_arrival = {"job": job, "value": random_value}

    # Increased arrival delay to be closer to processing time
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

class LogReporter(Reporter): #TODO: generalize log reporter for all patterns
    
    def __init__(self):
        
        self.df = pd.DataFrame(columns=['case_id', 'start_time', 'end_time', 'activity', 'value'])

        self.job_rewards = dict()
        self.arrival_times = dict()
        self.start_queue_times = dict()
        self.start_handling_times = dict()
        self.end_handling_times = dict()

        self.total_reward = 0
        self.total_wait_time = 0

    def callback(self, timed_binding):
        (binding, time, event) = timed_binding
        
        if event.get_id() == "job_arrival":
            job_id = binding[0][1].value["job"]
            value = binding[0][1].value["value"]
            start_time = time
            end_time = time+5
            df_entry = [job_id, start_time, end_time, event.get_id(), value]

            self.arrival_times[job_id] = start_time
            self.start_queue_times[job_id] = end_time

            # Add the event to the dataframe
            self.df.loc[len(self.df)] = df_entry

        elif event.get_id() == "job_handling":
            job_id = binding[1][1].value["job"]
            value = binding[1][1].value["value"]
            start_time = time
            end_time = time+7
            df_entry = [job_id, start_time, end_time, event.get_id(), value]

            self.start_handling_times[job_id] = start_time
            self.end_handling_times[job_id] = end_time
            self.job_rewards[job_id] = value

            self.total_wait_time += time - self.start_queue_times[job_id]
            self.total_reward += value
            
            # Add the event to the dataframe
            self.df.loc[len(self.df)] = df_entry

    def total_reward(self):
        return self.total_reward
    
    def mean_waiting_time(self):
        return self.total_wait_time / len(self.end_handling_times)
    
    def total_processing_time(self):
        return 7*len(self.end_handling_times)
    
    def total_idle_time(self):
        return production_line.clock - self.total_processing_time()
    
    def to_excel(self):
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.df.to_excel(f"priority_log_{current_date}.xlsx", index=False)

    def log(self):
        return self.df

# After simulation, print some statistics
visualize = False
simulate = True
log = True

if visualize:
    v = Visualisation(production_line)
    v.show()

if simulate:
    simtime = 10000
    pattern_reporter = PatternReporter(production_line, ["value"], ["priority"], simulation_name="priority")
    progress_reporter = ProgressReporter(simtime, print_interval=0.1)
    production_line.simulate(simtime, [pattern_reporter, progress_reporter])
    
    #log_df = log_reporter.log()
    pattern_df = pattern_reporter.get_state_df()

    # Log reporter
    #print("\nEvent Log:")
    #print(log_df.head())

    # State reporter
    print("\nProcess States:")
    print(pattern_df.head())

    # Print simulation statistics
    #print("--------------------------------")
    #print("\nSimulation Statistics:")

    #print(f"Total reward: {log_reporter.total_reward}")
    #print(f"Mean waiting time: {log_reporter.mean_waiting_time():.2f}")
    #print(f"Total processing time: {log_reporter.total_processing_time():.2f}")
    #print(f"Total idle time: {log_reporter.total_idle_time():.2f}")
    
    # Save logs to excel
    if log:
        #log_reporter.to_excel()
        pattern_reporter.to_excel()
        print("Saved to excel")


