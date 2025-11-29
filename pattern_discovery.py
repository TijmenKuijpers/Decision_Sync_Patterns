import pandas as pd
import numpy as np
from datetime import datetime

class PatternDiscovery:
    """
    A class for analyzing decision synchronization patterns from decision trees.
    """
    
    def __init__(self, clf, X, pn, tg, pattern_types=["priority", "blocking", "hold-batch", "choice"], leaf_samples_threshold=20, leaf_gini_threshold=0.2, gini_decrease_threshold=0.05):
        """
        Initialize the PatternDiscovery object.
        
        Parameters:
        clf: DecisionTreeClassifier - trained decision tree
        X: DataFrame - feature matrix
        pattern_types: list - types of patterns to analyze
        """
        self.pn = pn
        self.tg = tg
        self.clf = clf
        self.X = X
        self.pattern_types = pattern_types
        self.tree_ = clf.tree_
        self.feature_names = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else [f'feature_{i}' for i in range(self.tree_.n_features)]
        self.leaf_samples_threshold = leaf_samples_threshold
        self.leaf_gini_threshold = leaf_gini_threshold
        self.gini_decrease_threshold = gini_decrease_threshold
       
        # Pattern feature comparison operators
        self.direction_dict = {"priority": ["<=", ">"], 
                               "blocking": [">"], 
                               "hold-batch": ["<="], 
                               "choice": ["<="]} 

        # Store results
        self.results = {}
    
    def _get_node_path(self, target_node_id):
        """
        Get the path from root to a specific node.
        
        Parameters:
        target_node_id: int - target node ID
        
        Returns:
        list - list of node IDs from root to target
        """
        def find_path_recursive(node_id, path):
            if node_id == target_node_id:
                return path + [node_id]
            
            if self.tree_.children_left[node_id] != -1:  # Not a leaf
                # Try left child
                left_path = find_path_recursive(self.tree_.children_left[node_id], path + [node_id])
                if left_path:
                    return left_path
                
                # Try right child
                right_path = find_path_recursive(self.tree_.children_right[node_id], path + [node_id])
                if right_path:
                    return right_path
            
            return None
        
        return find_path_recursive(0, [])  # Start from root (node 0)
    
    def _calculate_gini(self, class_counts): #TODO: check if there is no function in clf for this
        """
        Calculate Gini impurity for a node.
        
        Parameters:
        class_counts: array - array of class counts
        
        Returns:
        float - Gini impurity
        """
        total = np.sum(class_counts)
        if total == 0:
            return 0
        
        probabilities = class_counts / total
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def analyze_constrained_patterns(self, pattern_type):
        """
        Analyze decision tree to extract pattern-specific splitting criteria for constrained classes.
        
        Parameters:
        pattern_type: str - type of pattern to analyze
        samples_threshold: int - minimum number of samples (default 20)
        gini_threshold: float - minimum gini decrease threshold (default 0.05)
        
        Returns:
        DataFrame with columns: feature, comparison_operator, value, true_false, samples, gini_impurity, gini_decrease
        """
        
        # Find leaf nodes that predict constrained classes
        constrained_leaves = []
        for node_id in range(self.tree_.node_count):
            if self.tree_.children_left[node_id] == -1:  # Leaf node
                # Get predicted class for this leaf
                class_counts = self.tree_.value[node_id][0]
                predicted_class_idx = np.argmax(class_counts)
                predicted_class = self.clf.classes_[predicted_class_idx]

                # Check if there is a guard in place
                if str(predicted_class) == "False":

                    # Only add leaf if it meets sample threshold, gini threshold
                    leaf_samples = self.tree_.n_node_samples[node_id]
                    leaf_gini = self._calculate_gini(self.tree_.value[node_id][0])
                    if leaf_samples  >= self.leaf_samples_threshold and leaf_gini < self.leaf_gini_threshold:

                        constrained_leaves.append(node_id)

        # Extract splitting criteria for each constrained leaf
        splitting_criteria = []
        
        for leaf_id in constrained_leaves:
            # Get path from root to leaf
            path = self._get_node_path(leaf_id)
            
            # Extract splitting criteria along the path
            for i, node_id in enumerate(path[:-1]):  # Exclude the leaf node itself
                if self.tree_.children_left[node_id] != -1:  # Not a leaf node
                    
                    # Check if comparison operator is in direction_dict
                    feature_idx = self.tree_.feature[node_id]
                    feature_name = self.feature_names[feature_idx]
                    next_node_id = path[i + 1]
                    is_true_branch = (next_node_id == self.tree_.children_left[node_id])
                    comparison_operator = '<=' if is_true_branch else '>'
                    allowed_operators = self.direction_dict[pattern_type]

                    if comparison_operator not in allowed_operators:
                        continue

                    # Record splitting criteria information
                    threshold = self.tree_.threshold[node_id]
                    
                    current_samples = self.tree_.n_node_samples[node_id] # Get samples and gini for current node
                    current_gini = self._calculate_gini(self.tree_.value[node_id][0])
                    next_gini = self._calculate_gini(self.tree_.value[next_node_id][0])

                    gini_decrease = current_gini - next_gini # Calculate gini decrease
                    if gini_decrease < self.gini_decrease_threshold:
                        continue

                    leaf_samples = self.tree_.n_node_samples[leaf_id] # Get samples and gini for leaf node
                    leaf_gini = self._calculate_gini(self.tree_.value[leaf_id][0])
                    predicted_event = self.clf.classes_[np.argmax(self.tree_.value[leaf_id][0])] # Get predicted event for leaf node
                    
                    # Add to splitting criteria
                    splitting_criteria.append({
                        'transition': self.tg,
                        'pattern': pattern_type,
                        'feature': feature_name,
                        'comparison_operator': comparison_operator,
                        'value': threshold,
                        'samples': current_samples,
                        'gini_impurity': current_gini,
                        'gini_decrease': gini_decrease,
                        'leaf_samples': leaf_samples,
                        'leaf_gini': leaf_gini,
                        'predicted_event': predicted_event,
                    })

        # Convert to DataFrame
        empty_df = pd.DataFrame(columns=['transition', 'pattern', 'feature', 'comparison_operator', 'value', 
                                          'samples', 'gini_impurity', 'gini_decrease', 'leaf_samples', 
                                          'leaf_gini', 'predicted_event'])
        if splitting_criteria:
            df = pd.DataFrame(splitting_criteria)

            # For constraints leading to same event, keep one with highest gini decrease
            df = df.sort_values('gini_decrease', ascending=False) # Other conceivably good criteria: keep closest to leave node
            df = df.drop_duplicates(subset=['feature', 'predicted_event'])
            
            # For priority and hold-batch patterns, ensure that there are exactly two splitting criteria
            if pattern_type == "priority" or pattern_type == "hold-batch":
                if len(df) != 2:
                    print(f"Warning: pattern_type '{pattern_type}' expects exactly two splitting criteria, but found {len(df)}.")
                    
                    return empty_df

            return df
        
        else:
            return empty_df
    
    def print_constrained_pattern_summary(self, constrained_patterns_df, pattern_type):
        """
        Print a summary of the constrained patterns analysis.
        
        Parameters:
        constrained_patterns_df: DataFrame - output from analyze_constrained_patterns
        pattern_type: str - type of pattern analyzed
        """
        if len(constrained_patterns_df) == 0:
            print(f"No {pattern_type} constrained patterns found for transition {self.tg}.")
            return
        
        print(f"\n=== {pattern_type.upper()} Constrained Patterns Summary ===")
        print(f"Total splitting criteria found: {len(constrained_patterns_df)}")
        
        # Group by feature
        feature_groups = constrained_patterns_df.groupby('feature')
        print(f"\nFeatures involved in {pattern_type} constrained patterns for transition {self.tg}:")
        for feature, group in feature_groups:
            print(f"  {feature}: {len(group)} criteria")
            for _, row in group.iterrows():
                print(f"    {row['comparison_operator']} {row['value']:.3f} - "
                      f"Samples: {row['samples']}, Leaf samples: {row['leaf_samples']}, Gini decrease: {row['gini_decrease']:.4f}")  
    