from simpn.simulator import SimProblem, SimToken, SimVarQueue, SimVar
from simpn.reporters import Reporter
from simpn.utils import seds

import copy
import inspect
import pandas as pd
import numpy as np

class SimPattern(SimProblem):

    def simulate(self, duration, reporter=None):
        """
        Executes a simulation run for the problem for the specified duration.
        A simulation run executes events until this is no longer possible, or until the specified duration.
        If at any moment multiple events can happen, one is selected at random.
        At the end of the simulation run, the problem is in the state that is the result of the run.
        If the reporter is set, its callback function is called with parameters (binding, time, event), and the result of the behavior of the event.
        each time a event happens (for a binding at a moment in simulation time).

        :param duration: the duration of the simulation.
        :param reporter: a class that implements a callback function, which is called each time a event happens. reporter can also be a list of reporters, in which case the callback function of each reporter is called.
        """
        active_model = True
        while self.clock <= duration and active_model:
            bindings = self.bindings()
            if len(bindings) > 0:
                
                timed_binding = self.binding_priority(bindings)
                timed_binding_copy = (copy.deepcopy(timed_binding[0]), timed_binding[1], timed_binding[2])
                # Watch out! The binding of queues will change after firing, resulting in incorrect reporting! Use deepcopy to mitigate
                # TODO: find a better way to handle this
                # print("timed binding: ", timed_binding)
                result = self.fire_result(timed_binding)
                #print("timed binding after firing: ", timed_binding)

                if reporter is not None:
                    if type(reporter) == list:
                        for r in reporter:
                            r.callback(timed_binding_copy, result)
                    else:
                        reporter.callback(timed_binding_copy, result)
                
            else:
                active_model = False

    def fire_result(self, timed_binding):
        """
        Fires the specified timed binding.
        Binding is a tuple ([(place, token), (place, token), ...], time, event)
        Return the result of the behavior of the event.
        """
        (binding, time, event) = timed_binding

        variable_assignment = []
        for (place, token) in binding:
            # remove tokens from incoming places
            place.remove_token(token)
            # assign values to the variables on the arcs
            variable_assignment.append(token.value)

        # calculate the result of the behavior of the event
        try:
            result = event.behavior(*variable_assignment)
        except Exception as e:
            raise TypeError("Event " + str(event) + ": behavior function generates exception for values " + str(variable_assignment) + ".") from e
        if self._debugging:
            if type(result) != list:
                raise TypeError("Event " + str(event) + ": behavior function does not generate a list for values " + str(variable_assignment) + ".")
            if len(result) != len(event.outgoing):
                raise TypeError("Event " + str(event) + ": behavior function does not generate as many values as there are output variables for values " + str(variable_assignment) + ".")
            i = 0
            for r in result:
                if r is not None:
                    if isinstance(event.outgoing[i], SimVarQueue):
                        if not isinstance(r, list):
                            raise TypeError("Event " + str(event) + ": does not generate a queue for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                    else:
                        if not isinstance(r, SimToken):
                            raise TypeError("Event " + str(event) + ": does not generate a token for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                        if not (type(r.delay) is int or type(r.delay) is float):
                            raise TypeError("Event " + str(event) + ": does not generate a numeric value for the delay of variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                        if not (type(r.time) is int or type(r.time) is float):
                            raise TypeError("Event " + str(event) + ": does not generate a numeric value for the time of variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                i += 1

        for i in range(len(result)):
            if result[i] is not None:
                if isinstance(event.outgoing[i], SimVarQueue):
                    event.outgoing[i].add_token(result[i])
                else:
                    if result[i].time > 0 and result[i].delay == 0:
                        raise TypeError("Deprecated functionality: Event " + str(event) + ": generates a token with a delay of 0, but a time > 0, for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ". It seems you are using the time of the token to represent the delay.")
                    token = SimToken(result[i].value, time=self.clock + result[i].delay)
                    event.outgoing[i].add_token(token)

        return result
    
    def advance_clock(self):
        """
        Advances the clock to the earliest time at which there is a new timed binding.
        """
        min_enabling_time = None

        # find the earliest enabling time for an event's incoming markings
        timings = dict()
        for ev in self.events:
            smallest_next = []
            skip = False
            added = False
            
            # identify when the earlier token could be used from places 
            # of the event; note: place.marking may not be ordered by
            # time, need to check for each place and otherwise walk the 
            # place's marking
            for place in ev.incoming:
                try:
                    if place.marking[0].time > self.clock:
                        smallest_next.append(place.marking[0].time) # Ensure going forward in time instead of getting stuck in constrained events
                        added = True
                except:
                    skip = True
            
            if (skip or not added):
                timings[ev] = 0
                continue
            
            # only keep the latest of the set of early tokens across places
            smallest_largest = max(smallest_next)
            timings[ev] = smallest_largest

            #if (smallest_largest == 0):
            #   continue

            # keep track of the smallest next possible clock
            if (smallest_largest is not None) \
                and (min_enabling_time is None \
                        or smallest_largest < min_enabling_time):
                min_enabling_time = smallest_largest 

        # timed bindings are only enabled if they have time <= clock
        # if there are no such bindings, set the clock to the earliest time 
        # at which there are
        if min_enabling_time is not None and min_enabling_time > self.clock:
            self.clock = min_enabling_time

class BehaviorEventLogReporter(Reporter):

    def __init__(self, sim_problem, filename, separator=","):
        self.sim_problem = sim_problem
        self.filename = filename
        self.function_report = {"event": [], "start_time": [], "completion_time": []}
        self.separator = separator

        for place in self.sim_problem.places:
            self.function_report[place.get_id()+"<incoming>"] = []
            self.function_report[place.get_id()+"<outgoing>"] = []

    def callback(self, timed_binding, result):
        (binding, time, event) = timed_binding        

        print("|event: ", event.get_id(), "|time: ", time, "|incoming: ", binding, "|result: ", result,"|")
        print("")

        # Add values for columns that have been associated with a value in this callback
        self.function_report["event"].append(event.get_id())
        self.function_report["start_time"].append(round(time, 2))
        
        # Use the longest delay as completion time for the full event
        delays = []
        for i in range(len(result)):
            try:
                delays.append(result[i].delay)
            except:
                continue
        max_delay = max(delays) if len(delays) > 0 else 0
        self.function_report["completion_time"].append(round(time+max_delay,2)) #[result[i].delay for i in range(len(result) if type(result[i]) != SimVarQueue else 0)]), 2)) # TODO: see how to get the delay in a clean way

        # Store incoming and outgoing values
        for i, place in enumerate(event.incoming):
            if ".queue" in place.get_id():
                self.function_report[place.get_id()[:-6]+"<incoming>"].append(binding[i][1].value)
            else:
                self.function_report[place.get_id()+"<incoming>"].append(binding[i][1].value)
        
        for i, place in enumerate(event.outgoing):

            if ".queue" in place.get_id():
                self.function_report[place.get_id()[:-6]+"<outgoing>"].append(result[i])
            else:
                self.function_report[place.get_id()+"<outgoing>"].append(result[i].value) # TODO: see how to get the outgoing values in a clean way
        
        # Add None values for columns that have not been associated with a value in this callback
        for column_name in self.function_report.keys():
            if len(self.function_report[column_name]) != len(self.function_report["event"]):
                self.function_report[column_name].append(None)

    def save_report(self):
        function_report_df = pd.DataFrame(self.function_report)
        function_report_df.to_csv(self.filename, sep=self.separator, index=False)

class GuardedAlignment:
    """
    A class that returns the guarded alignment of traces to a binding event log.
    A binding event log consists of one row for each transition that fired, 
    where each row contains the label of the transition that fired, the time at which it fired, and the binding of variables for which it was fired.
    It can be produced using the `reporters.BindingEventLogReporter` class.
    The guarded alignment is a replay of each event in the event log. An event can be replayed, if it can be fired in the current marking of the Petri net.
    Note that, due to randomness, matching of tokens from the event to the tokens in the net may not be possible based on equivalence. Consequently, matching will be based on similarity and the 'most similar' token will be matched.
    The output is an alignment table, with columns for the event label, the time of the event, an aligned column that is true or false depending on whether the event was aligned,
    and columns for variable values at the time just before firing the event. The variable values are calculated by functions that are provided as parameters.
    If the event can be replayed, a row will be included in the alignment table with aligned==True.
    If the event cannot be replayed, a row will be included in the alignment table for each transition that is enabled with aligned==False.
    In any case the binding will be fired, removing all tokens for which there is an equivalent in the current marking of the net and producing the output token.
    """
    INITIAL_STATE = "INITIAL_STATE"

    def __init__(self, sim_problem, event_log, separator=",", event_column="event", start_time_column="start_time", end_time_column="completion_time"):
        """
        Initializes the GuardedAlignment class with a SimProblem and an event log.

        :param sim_problem: An instance of SimProblem containing the process model.
        :param event_log: The filename of an event log.
        :param separator: The separator used in the event log file (default is comma).
        :param event_column_label: The name of the column in the event log that contains the event labels (default is "event").
        :param time_column: The name of the column in the event log that contains the event timestamps (default is "time").
        """
        self.sim_problem = sim_problem
        self.event_log = event_log
        self.separator = separator
        self.events = [] # a tuple (event, start_time, completion_time, log_binding, log_result), where log_binding is a dictionary of variable -> value and log_result is a dictionary of variable -> value

        # read the event log from file.
        # check if the event log is valid, meaning:
        # - it contains the event_column and the time_column
        # - each of the remaining columns is a variable in the sim_problem
        with open(event_log, 'r') as file:
            header = file.readline().strip().split(separator)
            if event_column not in header:
                raise ValueError(f"Event log does not contain the event column '{event_column}'")
            if start_time_column not in header:
                raise ValueError(f"Event log does not contain the time column '{start_time_column}'")
            if end_time_column not in header:
                raise ValueError(f"Event log does not contain the time column '{end_time_column}'")
            for var in [col for col in header if col != event_column and col != start_time_column and col != end_time_column]:
                if var.endswith("<incoming>") or var.endswith("<outgoing>"):
                    var = var[:-10]
                if sim_problem.var(var) is None:
                    raise ValueError(f"Variable '{var}' in event log is not a variable in the SimProblem")

            index2label = {i: col for i, col in enumerate(header) if col != event_column and col != start_time_column and col != end_time_column}

            for line in file:
                if line.strip():
                    parts = line.strip().split(separator)
                    event = None
                    start_time = None
                    end_time = None
                    log_binding = {}
                    log_result = {}
                    for i in range(len(parts)):
                        if header[i] == event_column:
                            event = parts[i]
                        elif header[i] == start_time_column:
                            start_time = float(parts[i])
                        elif header[i] == end_time_column:
                            end_time = float(parts[i])
                        elif header[i].endswith("<incoming>"):
                            if parts[i]:
                                try:
                                    log_binding[index2label[i]] = eval(parts[i])
                                except:
                                    log_binding[index2label[i]] = parts[i]
                            else:
                                log_binding[index2label[i]] = None
                        elif header[i].endswith("<outgoing>"):
                            if parts[i]:
                                try:
                                    log_result[index2label[i]] = eval(parts[i])
                                except:
                                    log_result[index2label[i]] = parts[i]
                            else:
                                log_result[index2label[i]] = None
                    self.events.append((event, start_time, end_time, log_binding, log_result))
        
        # Sort events by start time
        self.events.sort(key=lambda x: x[1])

    def evaluate(self, binding, function):
        """
        Evaluated the given function, with the parameters of the function bounds to the corresponding values of the binding.
        """
        # Get the function signature
        sig = inspect.signature(function)
        # Create a dictionary of parameter names to values
        params = {}
        for k in sig.parameters:
            if k.endswith("_queue"):
                queue_content = self.sim_problem.var(k[:-6]).marking
                token_values = [token.value for token in queue_content]
                params[k] = token_values
            elif k.endswith("_queue_tokens"):
                # Pass actual tokens instead of just values
                queue_content = self.sim_problem.var(k[:-13]).marking
                params[k] = list(queue_content)  # Pass tokens
            else:
                params[k] = binding[k]
        # Call the function with the bound parameters
        return function(**params)

    def calculate_functions(self, model_bindings, log_event, functions):
        """
        Calculates the values of the functions for the given model bindings and log event.
        """
        function_results = []
        for model_binding in model_bindings:
            binding_dict = log_event[3].copy()
            for var, token in model_binding[0]:
                binding_dict[var.get_id()] = token.value
            function_results.append([self.evaluate(binding_dict, f) for f in functions])
        return function_results
    
    def alignment(self, functions=[]):
        """
        Computes the alignment between the event log and the process model and returns the alignment table.
        For each event in the log, the table contains either: (1) a row for the event with aligned==True, if the event can be fired; or (2) one row for each enabled model binding with aligned==False, in case the event cannot be fired.
        Each row also contains the values of the specified functions at the time just before firing the event. These functions can be specified as parameters.
        Each function must have arguments that refer to variable names in the SimProblem. If a function has an argument that ends on '_queue', the content of the corresponding variable is provided as a list of all its token values.

        :param functions: A list of functions to be executed on the current state of the model.
        :return: A list of tuples (event, time, aligned, function values)
        """
        alignment = []
        self.sim_problem.store_checkpoint(self.INITIAL_STATE)  # Store the initial state of the process model
        
        # we need to evaluate the functions on each model binding before firing the transition, otherwise we use the wrong values (the ones after the decision was already made)
        for i, log_event in enumerate(self.events):
            
            model_bindings = self.sim_problem.bindings()
            
            # evaluate the functions on each model binding
            function_results = self.calculate_functions(model_bindings, log_event, functions)

            # when bindings are enabled before log event, add them to the allignment as False
            while log_event[1] > self.sim_problem.clock:
                model_bindings = self.sim_problem.bindings()
                function_results = self.calculate_functions(model_bindings, log_event, functions)
                for j in range(len(model_bindings)):
                        alignment_row = [model_bindings[j][2].get_id(), self.sim_problem.clock, False] + function_results[j]
                        if alignment_row not in alignment:
                            alignment.append(alignment_row)
                self.sim_problem.advance_clock() # step over this clock time untill new bindings are enabled
            
            # there is a matching binding if the event with the same label can be fired in the log
            log_binding, log_result, delay = self.reconstruct_binding_result(log_event)
            matching_binding_exists = len([event for (_, _, event) in model_bindings if event.get_id() == log_event[0]]) > 0
        
            # fire the log binding if the log event start time is now or in the past
            if self.sim_problem.clock >= log_event[1]:
                used_tokens = self.fire_log_binding(log_event[0], log_binding, log_result, delay) 

            # if a matching binding exists, add the fired event, the firing time (which is the current model clock), and True to the alignment
            if matching_binding_exists:
                alignment.append([log_event[0], self.sim_problem.clock, True] + function_results[0])  # TODO: we take the result of the first model binding, not sure if that is correct
            
            # if a matching binding does not exist, add the log event, the firing time, and False to the alignment
            else: 
                alignment.append([log_event[0], self.sim_problem.clock, False] + function_results[0])
            
            # if there are unmatched model bindings left, add them to the alignment as False
            if (len(model_bindings) > 1 and i + 1 < len(self.events) and self.events[i + 1][1] > log_event[1]):
                
                for j in range(len(model_bindings)):
                    if str(used_tokens[0].value) in str(model_bindings[j][0]): # If any used tokens are in the model binding, skip it
                        continue
                    
                    alignment_row = [model_bindings[j][2].get_id(), self.sim_problem.clock, False] + function_results[j]
                    if alignment_row not in alignment:
                        alignment.append(alignment_row)
        
        self.sim_problem.restore_checkpoint(self.INITIAL_STATE)  # Restore the initial state after generating traces
        return alignment
        
    def reconstruct_binding_result(self, event):
        """
        Remove <incoming> and <outgoing> suffixes from binding keys.
        """
        (_, start_time, complete_time, log_binding, log_result) = event
        # Remove <incoming> and <outgoing> suffixes from binding keys
        filtered_binding = {}
        filtered_result = {}
        for key, value in log_binding.items():
            if "<incoming>" in key:
                filtered_binding[key[:key.index("<")]] = value
        for key, value in log_result.items():
            if "<outgoing>" in key:
                filtered_result[key[:key.index("<")]] = value

        log_binding = filtered_binding
        log_result = filtered_result# (event, start_time, end_time, binding)
        delay = complete_time-start_time
        
        return log_binding, log_result, delay
        
    def fire_log_binding(self, event, log_binding, log_result, delay):
        """
        Fires the given log binding on the process model as follows:
        - for each incoming variable to the event, find the token with the closest value and remove that,
          where the closest value is determined by the mean string edit similarity on the variable's token values.
        - construct a timed binding from the log binding and use that to fire the transition,
          where the timed binding is constructed by setting the value for each incoming variable to the transition to the value of the log_binding.
        """
        model_transition = self.sim_problem.transition(event)
        variable_assignment = [] # a list of token values constructed from the log_binding
        tokens = []
        for var in model_transition.incoming:
            # find the token in the model that is closest to the value and remove it

            if ".queue" in var.get_id():
                log_binding_value = log_binding.get(var.get_id()[:-6])
            else:
                log_binding_value = log_binding.get(var.get_id())

            token = self.closest_token(var, log_binding_value) 
            tokens.append(token)

            var.remove_token(token)

            # instead, construct the assignment from the log binding
            variable_assignment.append(log_binding_value)
        #fire the transition and process the result
        result_model = model_transition.behavior(*variable_assignment)

        for i, var in enumerate(model_transition.outgoing):
            if result_model[i] is not None:
                if isinstance(model_transition.outgoing[i], SimVarQueue):
                    queue_tokens = self.log_queue_parser(log_result[var.get_id()[:-6]])
                    model_transition.outgoing[i].add_token(queue_tokens)
                else:
                    token = SimToken(log_result[var.get_id()], time=self.sim_problem.clock+delay) # TODO: add generated delay
                    model_transition.outgoing[i].add_token(token)
        
        return tokens


    def log_queue_parser(self, log_queue):
        """
        Splits the log_queue string at each ',' and removes the brackets.
        Returns a list of queue elements as strings (stripped of whitespace).
        """
        # Remove brackets
        log_queue_str = str(log_queue).strip()
        if log_queue_str.startswith("[") and log_queue_str.endswith("]"):
            log_queue_str = log_queue_str[1:-1]
        # Split by comma and strip whitespace
        elements = [elem.strip() for elem in log_queue_str.split(",") if elem.strip()]
        
        queue_tokens = []
        for element in elements:
            # Parse element of form "...@time"
            value_str, time_str = element.rsplit("@", 1)
            token = SimToken(value_str, time=float(time_str))
            queue_tokens.append(token)
        
        return queue_tokens
    
    def closest_token(self, var: SimVar, value):
        """
        Finds the token in the model's incoming variables that is closest to the given value.
        The closeness is determined by the mean string edit similarity.
        """

        str_value = str(value)
        best_match = None
        best_similarity = -1.0
        for token in var.marking:
            similarity = seds(str_value, str(token.value))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = token
        return best_match

    def save_alignment(self, alignment, filename, functions):
        """
        Saves the alignment to a CSV file.
        """
        header = ["Event", "Time", "Guard alligned"]
        for function in functions:
            header.append(function)
        alignment_df = pd.DataFrame(alignment, columns=header)
        alignment_df = alignment_df[alignment_df["Event"] != "timer"]
        alignment_df.to_excel(f"{filename}.xlsx", index=False)
        print(f"Alignment result saved to: {filename}.xlsx")
