import inspect
from simpn.simulator import SimToken, SimVar, SimVarQueue
from simpn.reporters import EventLogReporter
from tempfile import NamedTemporaryFile
from simpn.utils import seds
import pandas as pd
import numpy as np
import ast

class Conformance:
    """
    A class that can calculate the conformance of an event log, produced via the EventLogReporter, to a process model, specified as a SimProblem with BPMN prototypes.
    It has methods to calculate the fitness and the precision metrics.
    Fitness and precision are calculated based on the edit distance similarity between traces from the event log and the process model.
    The edit distance similarity between a trace t from the event log and a trace p from the process model is: the lowest number of insertions, deletions, and substitutions needed to transform t into p, divided by max(length of p, length of t).
    The fitness of a trace t from the event log is the highest edit distance similarity between t and any trace p from the process model.
    The fitness of the event log is the average fitness of all traces in the event log.
    The precision is analogously computed as the average fitness of all traces in the process model with respect to the event log.
    Preconditions:
    - we assume that events are in the log in the order of their occurrence in time.
    - the sim_problem must be in its initial state when the Conformance class is instantiated.
    """

    def __init__(self, sim_problem, event_log, separator=",", case_id_column="case_id", task_column="task", sample_duration=1000):
        """
        Initializes the Conformance class with a SimProblem and an event log.

        :param sim_problem: An instance of SimProblem containing the process model.
        :param event_log: The filename of an event log.
        :param separator: The separator used in the event log file (default is comma).
        :param case_id_column: The name of the column in the event log that contains the case IDs (default is "case_id").
        :param task_column: The name of the column in the event log that contains the task labels (default is "task").
        :param sample_duration: The time the simulator will be run to generate traces to calculate the fitness/ precision over (default is 1000).
        """
        self.INITIAL_STATE = "initial state"
        self.sim_problem = sim_problem
        self.sim_problem.store_checkpoint(self.INITIAL_STATE)  # Store the initial state of the process model
        self.event_log = event_log
        self.separator = separator
        self.case_id_column = case_id_column
        self.task_column = task_column
        self.sample_duration = sample_duration
        self.traces_sampled_from_process = self.generate_traces(duration=sample_duration)
        self.sim_problem.restore_checkpoint(self.INITIAL_STATE)  # Restore the initial state after generating traces
        self.traces_extracted_from_log = self.extract_traces(event_log, separator=separator, case_id_column=case_id_column, task_column=task_column)
    
    def extract_traces(self, event_log, separator=",", case_id_column="case_id", task_column="task"):
        """
        Extracts traces from the event log.
        The result is a dictionary of trace -> frequency of occurrence, where each trace is a list of task labels.

        :return: A dictionary of traces with the frequency of their occurrence.
        """
        with open(event_log, 'r') as file:
            header = file.readline().strip().split(separator)
            case_id_index = header.index(case_id_column)
            task_index = header.index(task_column)

            # A dictionary case_id -> list of tasks
            cases = {}

            for line in file:
                if line.strip():
                    parts = line.strip().split(separator)
                    case_id = parts[case_id_index]
                    task = parts[task_index]

                    if case_id not in cases:
                        cases[case_id] = []
                    cases[case_id].append(task)

            # Convert cases to traces
            traces = {}
            for case_id, tasks in cases.items():
                trace = tuple(tasks)
                if trace not in traces:
                    traces[trace] = 0
                traces[trace] += 1

        return traces

    def generate_traces(self, duration=1000):
        """
        Generates traces from the SimProblem's process model.
        The result is a dictionary of trace -> frequency of occurrence, where each trace is a list of task labels.

        :return: A dictionary of traces with the frequency of their occurrence.
        """

        # We can generate traces by simulating the process model, storing the event log in a temporary file, and then extracting the traces from that log.
        with NamedTemporaryFile() as temp_log:            
            reporter = EventLogReporter(temp_log.name, separator=self.separator)
            self.sim_problem.restore_checkpoint(self.INITIAL_STATE)
            self.sim_problem.simulate(duration, reporter)
            temp_log.flush()

            # Now extract traces from the temporary log file
            traces = self.extract_traces(temp_log.name, separator=self.separator)

            # Clean up the temporary file
            temp_log.close()

            return traces

    def edit_distance_similarity(self, trace1, trace2):
        """
        Calculates the edit distance similarity between two traces.
        The edit distance similarity is defined as the lowest number of insertions, deletions, and substitutions needed to transform trace1 into trace2, divided by max(length of trace1, length of trace2).

        :param trace1: A list representing the first trace.
        :param trace2: A list representing the second trace.
        :return: A float representing the edit distance similarity.
        """
        len1 = len(trace1)
        len2 = len(trace2)

        # Create a distance matrix
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            for j in range(len2 + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif trace1[i - 1] == trace2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                                   dp[i][j - 1] + 1,      # Insertion
                                   dp[i - 1][j - 1] + 1)
        edit_distance = dp[len1][len2]
        max_length = max(len1, len2)
        if max_length == 0:
            return 1.0
        return 1 - (edit_distance / max_length)  # Return similarity as a value

    def calculate_fit(self, from_traces, to_traces):
        """
        Calculates the fitness of a set of traces with respect to another set of traces.
        The fitness is defined as the average similarity of each trace in from_traces to the best matching trace in to_traces.
        :param from_traces: A dictionary of traces with their frequencies.
        :param to_traces: A dictionary of traces with their frequencies.
        :return: A float representing the fitness value.
        """
        fitness_values = []

        for from_trace, from_frequency in from_traces.items():
            max_similarity = 0.0
            for to_trace in to_traces.keys():
                similarity = self.edit_distance_similarity(from_trace, to_trace)
                if similarity > max_similarity:
                    max_similarity = similarity
            fitness_values.append(max_similarity * from_frequency)

        if not fitness_values:
            return 0.0
        return sum(fitness_values) / sum(from_traces.values())

    def calculate_fitness(self):
        """
        Calculates the fitness of the event log with respect to the process model.

        :return: A float representing the fitness value.
        """
        return self.calculate_fit(self.traces_extracted_from_log, self.traces_sampled_from_process)        
    
    def calculate_precision(self):
        """
        Calculates the precision of the event log with respect to the process model.
        :return: A float representing the precision value.
        """
        return self.calculate_fit(self.traces_sampled_from_process, self.traces_extracted_from_log)


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
        self.events = [] # a tuple (event, time, end_time, log_binding, log_result), where log_binding is a dictionary of variable -> value and log_result is a dictionary of variable -> value

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
                                    print("Error evaluating log binding: ", parts[i])
                                    log_binding[index2label[i]] = parts[i]
                            else:
                                log_binding[index2label[i]] = None
                        elif header[i].endswith("<outgoing>"):
                            if parts[i]:
                                try:
                                    log_result[index2label[i]] = eval(parts[i])
                                except:
                                    print("Error evaluating log result: ", parts[i])
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
        #for i, log_event in enumerate(log_events_at_time):
        for i, log_event in enumerate(self.events):
            
            model_bindings = self.sim_problem.bindings()
            print("--------------------------------")
            print("Considering log event: ", log_event[0], ", at time: ", self.sim_problem.clock)
            print("Current model bindings: ", model_bindings)
            # evaluate the functions on each model binding
            function_results = self.calculate_functions(model_bindings, log_event, functions)

            # when bindings are enabled before log event, add them to the allignment as False
            while log_event[1] > self.sim_problem.clock:
                #print("There are model bindings before log time: t=", log_event[1])
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
                #print("There is a log binding with a matching model binding")
                alignment.append([log_event[0], self.sim_problem.clock, True] + function_results[0])  # TODO: we take the result of the first model binding, not sure if that is correct
            
            # if a matching binding does not exist, add the log event, the firing time, and False to the alignment
            else:
                #print("There is a log binding with no matching model binding")
                alignment.append([log_event[0], self.sim_problem.clock, False] + function_results[0])
            
            # if there are unmatched model bindings left, add them to the alignment as False
            if (len(model_bindings) > 1 and i + 1 < len(self.events) and self.events[i + 1][1] > log_event[1]):
                #print("There are model bindings with no matching log binding")
                
                for j in range(len(model_bindings)):
                    if str(used_tokens[0].value) in str(model_bindings[j][0]): # If any used tokens are in the model binding, skip it
                        #print("Model binding already fired")
                        continue
                    
                    alignment_row = [model_bindings[j][2].get_id(), self.sim_problem.clock, False] + function_results[j]
                    if alignment_row not in alignment:
                        #print("Adding model binding to alignment")
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

            token = self.closest_token(var, log_binding_value) # TODO: make sure this works for queues
            tokens.append(token)

            #print("original marking: ", var.marking)
            var.remove_token(token)
            #print("marking after removal: ", var.marking)
            # instead, construct the assignment from the log binding
            variable_assignment.append(log_binding_value)
        #fire the transition and process the result
        result_model = model_transition.behavior(*variable_assignment)

        for i, var in enumerate(model_transition.outgoing):
            if result_model[i] is not None:
                if isinstance(model_transition.outgoing[i], SimVarQueue):
                    #print(type(result_model[i][0]))
                    queue_tokens = self.log_queue_parser(log_result[var.get_id()[:-6]])
                    #print("adding: ", queue_tokens)
                    model_transition.outgoing[i].add_token(queue_tokens)
                    #print("updated marking: ", model_transition.outgoing[i].marking)
                else:
                    token = SimToken(log_result[var.get_id()], time=self.sim_problem.clock+delay) # TODO: add generated delay
                    #print("adding: ", token)
                    model_transition.outgoing[i].add_token(token)
                    #print("updated marking: ", model_transition.outgoing[i].marking)
        
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