from sc_prototypes import SCOrder, SCDemand, SCStock
from simpn.simulator import SimProblem, SimToken
from simpn.visualisation import Visualisation
from simpn.reporters import FunctionEventLogReporter
import random
from itertools import combinations
from datetime import datetime
import pandas as pd
import sys
import os

# Add the Patterns directory to the path to import analysis_branch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_branch import GuardedAlignment

# Add the parent directory to the path to import pattern_reporter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pattern_reporter import PatternReporter, ProgressReporter

class SCMGameSimulation:
    """
    A class to run SCM game simulations with different parameterizations.
    """
    
    def __init__(self, sourcing, parameters, uncertainty, initial_stock=None):
        """
        Initialize the SCM game simulation with specific parameters.
        
        Parameters:
        sourcing: dict - sourcing levels for different components
        parameters: dict - lead times and other parameters
        uncertainty: dict - uncertainty settings for different processes
        initial_stock: dict - initial stock levels (optional)
        """
        self.sourcing = sourcing
        self.parameters = parameters
        self.uncertainty = uncertainty
        self.initial_stock = initial_stock or {"s1": 2, "s2": 4, "s3": 2, "d1": 3, "d2": 0, "d3": 0}
        
        # Initialize simulation problem
        self.SCgame = SimProblem()
        self.pattern_reporter = None
        
        # Initialize places and variables
        self._initialize_places()
        self._initialize_events()
    
    def _initialize_places(self):
        """Initialize all places and variables in the simulation."""
        # Add a timer to monitor time-related guards
        #log_timer = self.SCgame.add_var("log_timer")
        #self.SCgame.add_event([log_timer], [log_timer], behavior=lambda log_timer: [SimToken(0, delay=0.05)], name="timer")
        #log_timer.put(0)

        
        # Model the supply chain stock points
        self.source_pc = self.SCgame.add_var("source_pc")
        self.source_c = self.SCgame.add_var("source_c")
        self.source_gc = self.SCgame.add_var("source_gc")

        self.s1 = SCStock(self.SCgame, "s1")#, priority=lambda token: -token.value["priority"])
        self.s2 = SCStock(self.SCgame, "s2")
        self.s3 = SCStock(self.SCgame, "s3")

        self.d1 = SCDemand(self.SCgame, "d1")
        self.d2 = SCDemand(self.SCgame, "d2")
        self.d3 = SCDemand(self.SCgame, "d3")

        # Initialize the sourcing
        #for i in range(self.sourcing["source_pc"]):
        self.source_pc.put({"action": "source phone case", "priority": 0})
        #for i in range(self.sourcing["source_c"]):
        self.source_c.put({"action": "source chip"})
        #for i in range(self.sourcing["source_gc"]):
        self.source_gc.put({"action": "source game case"})

        # Initialize the stock
        for i in range(self.initial_stock["s1"]):
            self.s1.put({"action": "order phone case", "priority": 0})
        for i in range(self.initial_stock["s2"]):
            self.s2.put({"action": "order chip"})
        for i in range(self.initial_stock["s3"]):
            self.s3.put({"action": "order game case"})
        for i in range(self.initial_stock["d1"]):
            self.d1.put(0)

        # Include a truck resource
        self.truck = self.SCgame.add_var("truck")
        self.truck.put({"truck_id": 1, "loading": False})

        # Select an action every day
        self.cturn = self.SCgame.add_var("turn")
        self.cturn.put({"demand": 0})  
        self.cturn.set_invisible()

        self.demand = self.SCgame.add_var("demand")
        self.demand.set_invisible()
    
    def _initialize_events(self):
        """Initialize all events in the simulation."""
        # Get the delay parameterization
        phone_case_lt = self.parameters["phone_case_lt"]
        chip_lt = self.parameters["chip_lt"]
        game_lt = self.parameters["game_lt"]
        phone_prod_lt = self.parameters["phone_prod_lt"]
        game_prod_lt = self.parameters["game_prod_lt"]

        # Get the uncertainty parameterization
        phone_case_uncertainty = self.uncertainty["phone_case_uncertainty"]
        chip_uncertainty = self.uncertainty["chip_uncertainty"]
        game_uncertainty = self.uncertainty["game_uncertainty"]
        phone_prod_uncertainty = self.uncertainty["phone_prod_uncertainty"]
        game_prod_uncertainty = self.uncertainty["game_prod_uncertainty"]

        # Control the flow of the supply chain
        self.o1 = SCOrder(self.SCgame, [self.source_pc], [self.source_pc, self.s1], "order phone case", 
                         behavior=lambda p: [SimToken(p, delay=0)],
                         outgoing_behavior=self.phone_order_outgoing_behavior, 
                         guard=None)#self._phone_order_guard) 
        self.o2 = SCOrder(self.SCgame, [self.source_c], [self.source_c, self.s2], "order chip", 
                         behavior = lambda p: [SimToken(p, delay=0)], 
                         outgoing_behavior=self.chip_order_outgoing_behavior, 
                         guard=None)#self._chip_order_guard)
        self.o3 = SCOrder(self.SCgame, [self.source_gc], [self.source_gc, self.s3], "order game case", 
                         behavior = lambda p: [SimToken(p, delay=0)],
                         outgoing_behavior=self.game_order_outgoing_behavior, 
                         guard=None)#self._game_order_guard)

        self.p1 = self.SCgame.add_event([self.s1, self.s2], [self.d1], name="prod phone",  
                         behavior=lambda s1, s2: [SimToken("ffil phone NL", delay=random.expovariate(1/phone_prod_lt) if phone_prod_uncertainty else phone_prod_lt)], 
                         guard=None)#self._phone_production_guard) # Produce phone cases
        self.p2 = self.SCgame.add_event([self.s2, self.s3], [self.d2], name="prod game", 
                         behavior=lambda s2, s3: [SimToken("ffil game NL", delay=random.expovariate(1/game_prod_lt) if game_prod_uncertainty else game_prod_lt)], 
                         guard=None)#self._game_production_guard) # Produce game cases
        self.t1 = self.SCgame.add_event([self.d1, self.truck], [self.d3, self.truck], name="trans phone", 
                         behavior=self._transportation,
                         guard=None)#self._transportation_guard) 

        self.ff1 = self.SCgame.add_event([self.demand, self.d1], [], lambda d, d1: [],guard=lambda d, d1: d=="ffil phone NL", name="ffill 1") #  Execute demand fulfillment
        self.ff2 = self.SCgame.add_event([self.demand, self.d2], [], lambda d, d2: [],guard=lambda d, d2: d=="ffil game NL", name="ffill 2")
        self.ff3 = self.SCgame.add_event([self.demand, self.d3], [], lambda d, d3: [],guard=lambda d, d3: d=="ffil phone DE", name="ffill 3")

        self.ff1.set_invisible()
        self.ff2.set_invisible()
        self.ff3.set_invisible()

        self.demand_event = self.SCgame.add_event([self.demand.queue, self.cturn], [self.demand.queue, self.cturn], self._demand_generator, guard=lambda p, c: c == "demand")
        self.demand_event.set_invisible()

    def phone_order_outgoing_behavior(self, p):
        """
        The outgoing behavior of the phone order event.
        """

        phone_case_lt = self.parameters["phone_case_lt"]
        phone_case_uncertainty = self.uncertainty["phone_case_uncertainty"]

        delay = random.expovariate(1/phone_case_lt) if phone_case_uncertainty else phone_case_lt

        return [SimToken({"action": "source phone case", "priority": 1 if random.random() < 0.33 else 0}, delay=delay), SimToken(p, delay=delay)]
    
    def chip_order_outgoing_behavior(self, p):
        """
        The outgoing behavior of the chip order event.
        """

        chip_lt = self.parameters["chip_lt"]
        chip_uncertainty = self.uncertainty["chip_uncertainty"]

        delay = random.expovariate(1/chip_lt) if chip_uncertainty else chip_lt

        return [SimToken({"action": "source chip"}, delay=delay), SimToken(p, delay=delay)]
    
    def game_order_outgoing_behavior(self, p):
        """
        The outgoing behavior of the game order event.
        """

        game_lt = self.parameters["game_lt"]
        game_uncertainty = self.uncertainty["game_uncertainty"]

        delay = random.expovariate(1/game_lt) if game_uncertainty else game_lt

        return [SimToken(p, delay=delay), SimToken(p, delay=delay)]
    

    def _transportation(self, d1_token, truck):
        # Collect the relevant information
        all_tokens = self.d1.queue.marking[0].value
        time_enabled = [token for token in all_tokens if token.time <= self.SCgame.clock]

        # Get the delay and uncertainty parameterization
        truck_lt = self.parameters["truck_lt"]
        truck_uncertainty = self.uncertainty["truck_uncertainty"]

        # Check if the truck should depart
        if len(time_enabled) > 0:
            delay_truck = 0
            truck["loading"] = True
            return [SimToken("ffil phone DE", delay=delay_truck), SimToken(truck, delay=delay_truck)]#delay=random.expovariate(1/truck_lt) if truck_uncertainty else truck_lt), SimToken(truck, delay=delay_truck)]
        elif len(time_enabled) == 0:
            delay_truck = 2
            truck["loading"] = False
            return [SimToken("ffil phone DE", delay=delay_truck), SimToken(truck, delay=delay_truck)]#delay=random.expovariate(1/truck_lt) if truck_uncertainty else truck_lt), SimToken(truck, delay=random.expovariate(1/truck_lt))]#delay_truck)]

    def _demand_generator(self, demand, turn):
        # List of hubs that can incur demand
        hubs = ["ffil game NL",
                "ffil phone NL",
                "ffil phone DE",]
            
        demand_tokens = demand
        demand_loc = random.choice(hubs) # Randomly choose hub that incurs demand
        demand_count = random.randint(1, 6) # Demand uniform between 1-6

        demand_tokens = [] # No backorders

        for count in range(demand_count): # Generate "count" tokens on demand location
            demand_token = SimToken(demand_loc, time=self.SCgame.clock)
            demand_tokens.append(demand_token)

        return [demand_tokens, SimToken("demand", delay=1)]

    def run_simulation(self, simtime=1000, visualize=False, log=True, file_save=None, pattern_types=None, analysis=False):
        """
        Run the simulation with the current parameterization.
        
        Parameters:
        simtime: int - simulation time
        visualize: bool - whether to show visualization
        log: bool - whether to save logs
        file_save: str - filename to save the log
        pattern_types: list - list of pattern types to report on (e.g., ["priority", "blocking", "hold-batch", "choice"])
        
        Returns:
        tuple - (state_df, filename) where filename is None if log=False
        """
        # Default pattern types if not specified
        if pattern_types is None:
            pattern_types = ["blocking", "hold-batch", "choice"]
        
        # Create reporter for state logging
        self.pattern_reporter = PatternReporter(
            SimProblem=self.SCgame,
            attributes=['priority'],  # Attributes to track for priority patterns
            pattern_types=pattern_types,
            skip_places=["log_timer", "truck", "turn", "demand"],
            skip_events=["timer", "_demand_generator", 'ffill 1', 'ffill 2', 'ffill 3', 
                        "ffill 1_constrained", "ffill 2_constrained", "ffill 3_constrained",
                        'order phone case<task:complete>', 'order chip<task:complete>', 'order game case<task:complete>'],
            simulation_name="scm_game"
        )
        #self.progress_reporter = ProgressReporter(simtime, print_interval=0.05)
        
        if visualize:
            v = Visualisation(self.SCgame)
            v.show()
        
        #pattern_df = self.pattern_reporter.get_state_df()

        # Save logs to excel
        if log:
            print(f"Simulating with parameters: {self.parameters}")
            print(f"Sourcing: {self.sourcing}")
            print(f"Uncertainty: {self.uncertainty}")
            print(f"Pattern types being reported: {pattern_types}")
        
            function_event_log_reporter = FunctionEventLogReporter(self.SCgame, file_save, separator=";")
            self.SCgame.simulate(simtime, [function_event_log_reporter])
            function_event_log_reporter.save_report()
            #self.pattern_reporter.to_excel(filename=file_save)
            print(f"Saved function event log to: {file_save}")

        if analysis:

            def is_max_enabled(queue):
                queue_values = [token.value["priority"] for token in queue]
                queue_enabled_values = [token.value["priority"] for token in queue if token.time <= self.SCgame.clock]

                if len(queue_values) > 0:
                    queue_max = max(queue_values)
                else:
                    queue_max = -1 # Not the same as queue_enabled_max
                
                if len(queue_enabled_values) > 0:
                    queue_enabled_max = max(queue_enabled_values)
                else:
                    queue_enabled_max = 0
                    
                return queue_max == queue_enabled_max

            def time_until_next_enabled(queue):
                waiting_tokens = [token for token in queue if token.time > self.SCgame.clock]
                
                if len(waiting_tokens) > 0:
                    return waiting_tokens[0].time - self.SCgame.clock
                else:
                    return 100000

            alignment = GuardedAlignment(self.SCgame, file_save, separator=";")
            result = alignment.alignment(functions=[lambda s3_queue: len(s3_queue), # Blocking
                                                   lambda source_pc_queue: max(token["priority"] for token in source_pc_queue) if source_pc_queue else 0, # Priority
                                                   lambda s1_queue: max(token["priority"] for token in s1_queue) if s1_queue else 0, 
                                                   lambda s1_queue_tokens: is_max_enabled(s1_queue_tokens),
                                                   lambda s1_queue_tokens: time_until_next_enabled(s1_queue_tokens),# Choice
                                                   lambda d1_queue_tokens: len([token for token in d1_queue_tokens if token.time <= self.SCgame.clock]), # Hold-batch
                                                   lambda d1_queue_tokens: time_until_next_enabled(d1_queue_tokens)])

            functions = ["s3_queue_length", "max_source_pc_value", "max_s1_value", "max_s1_value_is_enabled", "s1 time until next enabled", "d1 enabledqueue length", "d1 queue time until next enabled"]
            save_filename = f"scm_game_structural_alignment_"+file_save.split(".")[0]
            alignment.save_alignment(result, save_filename, functions)

# Run the default simulation
if __name__ == "__main__":
    # Parameterization
    sourcing = {"source_pc": 1, "source_c": 2, "source_gc": 1}
    parameters = {"phone_case_lt": 1, "chip_lt": 2, "game_lt": 1, "phone_prod_lt": 2, "game_prod_lt": 2, "truck_lt": 1}
    uncertainty = {"phone_case_uncertainty": True, "chip_uncertainty": True, "game_uncertainty": True, "phone_prod_uncertainty": False, "game_prod_uncertainty": False, "truck_uncertainty": False}
    pattern_types = ["priority", "blocking", "hold-batch", "choice"]
    # Create and run simulation
    nr_of_simulations = 10
    for i in range(nr_of_simulations):
        filename = f"scm_game_log_{i+1}.csv"
        print(f"Running simulation {i+1} of {nr_of_simulations}")
        simulation = SCMGameSimulation(sourcing, parameters, uncertainty)
        simulation.run_simulation(simtime=100, visualize=False, log=False, file_save=filename, pattern_types=pattern_types, analysis=True)
    print(f"Simulation {i+1} completed")
    print(f"{'='*50}")
    
    #filename = f"scm_game_log_prio_updated.xlsx"
    #simulation = SCMGameSimulation(sourcing, parameters, uncertainty)
    #pattern_df = simulation.run_simulation(simtime=1000, visualize=False, log=True, file_save=filename, pattern_types=pattern_types)
       

    #print("\nProcess States:")
    #print(pattern_df.head())
    #print(f"Saved to excel: {filename}")

