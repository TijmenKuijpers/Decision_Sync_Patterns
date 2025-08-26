import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree

from pattern_discovery import PatternDiscovery

class DataPreprocessor:
    """Class responsible for data loading and preprocessing operations."""
    
    def __init__(self):
        self.feature_drop_list = []
    
    def load_and_prepare_data(self, filename, pn, transition, pattern):
        """Load and prepare data for decision tree training."""
        # Read the state data
        df = pd.read_excel(filename)
        df = df[df['event_id'].isin([transition, f"{transition}_constrained"])]
        
        # Apply preprocessing
        #df = self._apply_general_cleaning(df, pn, transition, pattern)
        df = self._apply_pattern_specific_cleaning(df, pn, transition, pattern)

        # Save preprocessed dataframe to excel with timestamp

        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create safe transition name by replacing special characters
        #safe_transition = transition.replace('<', '_').replace('>', '_').replace(':', '_')
        #output_filename = f'preprocessed_{safe_transition}_{pattern}_{timestamp}.xlsx'
        #df.to_excel(output_filename, index=False)
        #print(f"Preprocessed data saved to: {output_filename}")

        # Create observation (X) and target (y) variables
        X, y = self._create_features_and_targets(df, transition)
        
        # Print data info
        #self._print_data_info(X)
        
        return X, y
    
    def _apply_general_cleaning(self, df, pn, transition, pattern):
        """Apply pattern-specific data cleaning and feature engineering."""
        # Drop columns containing 'log' in their names and time until enabled columns
        log_columns = [col for col in df.columns if 'log' in col.lower()]
        time_until_enabled_cols = [col for col in df.columns if col.endswith('_time_until_enabled')]
        self.feature_drop_list.extend(log_columns)
        self.feature_drop_list.extend(time_until_enabled_cols)

        # Scan for features with only one unique value
        for col in df.columns:
            if col not in ['time', 'event_id'] and 'time_until_enabled':# not in col and df[col].nunique() == 1:
                if col not in self.feature_drop_list:
                    #print(f"Column {col} dropped: only one unique value")
                    self.feature_drop_list.append(col)
        
        df = df.drop(self.feature_drop_list, axis=1)
        return df
    
    def _apply_pattern_specific_cleaning(self, df, pn, transition, pattern):
        """Apply pattern-specific data cleaning and feature engineering."""
        # Get all places in process from pn dictionary
        all_places = set()
        for transition_flows in pn.values():
            input_places, output_places = transition_flows
            all_places.update(input_places)
            all_places.update(output_places)

        all_places = list(all_places)

        if pattern == "priority":
            df = self._clean_priority_pattern(df, pn, transition, all_places)
        elif pattern == "blocking":
            df = self._clean_blocking_pattern(df, pn, transition, all_places)
        elif pattern == "hold-batch":
            df = self._clean_hold_batch_pattern(df, pn, transition, all_places)
        elif pattern == "choice":
            df = self._clean_choice_pattern(df, pn, transition, all_places)

        return df
    
    def _clean_priority_pattern(self, df, pn, transition, all_places):
        """Clean data for priority pattern."""
        
        print(df.columns)

        # Process construct valid places
        input_places = pn[transition][0]
        upstream_places = set()
        for t, (ins, outs) in pn.items():
            if any(out_place in input_places for out_place in outs):
                upstream_places.update(ins)
        valid_places = list(set(input_places) | upstream_places)
        
        outscope_places = [place for place in all_places if place not in valid_places]
        outscope_cols = [col for col in df.columns if any(outscope_place in col for outscope_place in outscope_places)]
        df = df.drop(outscope_cols, axis=1)
        
        # Create attribute ratio features
        min_max_cols = [col for col in df.columns if col.endswith(('_min', '_max'))]
        places = list(set([col.split('_')[0] for col in min_max_cols]))
        attributes = list(set([col.split('_')[0] for col in [col.split('_')[1] for col in min_max_cols]]))
    
        #print(places)
        #print(attributes)
        #print(df.columns)
        print(df.columns)

        for place in places: # Add a feature to check if the min or max are enabled
            if place not in pn[transition][0]:
                continue
            for attr in attributes:
                try:
                    df[f'{place}_{attr}_max_is_enabled'] = df[f'{place}_{attr}_enabled_max'] == df[f'{place}_{attr}_max']
                except:
                    print(f"Error creating {place}_{attr}_max_is_enabled")
                    continue
                try:
                    df[f'{place}_{attr}_min_is_enabled'] = df[f'{place}_{attr}_enabled_min'] == df[f'{place}_{attr}_min']
                except:
                    print(f"Error creating {place}_{attr}_min_is_enabled")
                    continue

        for i, col1 in enumerate(min_max_cols):
            for col2 in min_max_cols[i+1:]:
                # Extract place names from column names
                place1 = col1.split('_')[0]
                place2 = col2.split('_')[0]
                
                # Skip if both columns are from the same place
                if place1 == place2: #and place1 not in pn[transition][0]:
                    continue

                ratio_name = f'{col1}/{col2}'
                df[ratio_name] = df[col1] / df[col2].replace(0, 0.01) # Replace 0s with 0.01 to avoid division by zero
                df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], 0)

        # Remove highly correlated ratios
        ratio_features = [col for col in df.columns if '/' in col]
        if ratio_features:
            ratio_corr = df[ratio_features].corr().abs()
            upper = ratio_corr.where(np.triu(np.ones(ratio_corr.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            if to_drop:
                df = df.drop(to_drop, axis=1)

        # Drop irrelevant features -> keep only attribute based features
        time_cols = [col for col in df.columns if 'time_until_next_enabled' in col] # drops time until next enabled
        token_cols = [col for col in df.columns if 'token_count' in col] # drops (enabled) token counts
        non_ratio_cols = [col for col in df.columns if 
                            col not in ratio_features and 
                            col != 'event_id' and 
                            col != 'time' and 
                            'enabled_min' not in col and 
                            'enabled_max' not in col and
                            'min_is_enabled' not in col and
                            'max_is_enabled' not in col]
        df = df.drop(time_cols + token_cols + non_ratio_cols, axis=1)
        
        return df
    
    def _clean_blocking_pattern(self, df, pn, transition, all_places):
        """Clean data for blocking pattern."""
        # Drop irrelevant features -> keep only enabled token count and time until next enabled features
        time_cols = [col for col in df.columns if 'time_until_next_enabled' in col]
        token_cols = [col for col in df.columns if 'enabled_token_count' in col]
        max_cols = [col for col in df.columns if 'max' in col]
        min_cols = [col for col in df.columns if 'min' in col]
        
        # Process construct valid places
        output_places = pn[transition][1]
        outscope_places = [place for place in all_places if place not in output_places]
        outscope_cols = [col for col in df.columns if col.split('_')[0] in outscope_places]
        
        df = df.drop(time_cols + token_cols + max_cols + min_cols + outscope_cols, axis=1)
        return df
    
    def _clean_hold_batch_pattern(self, df, pn, transition, all_places):
        """Clean data for hold-batch pattern."""
        
        # Remove consecutive transportation events
        df['is_transport'] = df['event_id'] == transition #TODO this is not general for any process
        df['prev_transport'] = df['is_transport'].shift(1).fillna(False)
        df = df[~(df['is_transport'] & df['prev_transport'])]
        df = df.drop(['is_transport', 'prev_transport'], axis=1)
        
        # Drop irrelevant features -> keep only enabled token count and time until next enabled features
        token_count_cols = [col for col in df.columns if '_token_count' in col and not '_enabled_token_count' in col]
        max_cols = [col for col in df.columns if 'max' in col]
        min_cols = [col for col in df.columns if 'min' in col]
        
        # Process construct valid places
        input_places = pn[transition][0]
        outscope_places = [place for place in all_places if place not in input_places]
        
        outscope_cols = [col for col in df.columns if col.split('_')[0] in outscope_places]
        df = df.drop(token_count_cols + max_cols + min_cols + outscope_cols, axis=1)
        
        return df
    
    def _clean_choice_pattern(self, df, pn, transition, all_places):
        """Clean data for choice pattern."""
        # Drop irrelevant features -> keep only enabled token count and time until next enabled features
        token_count_cols = [col for col in df.columns if '_token_count' in col and not '_enabled_token_count' in col]
        max_cols = [col for col in df.columns if 'max' in col]
        min_cols = [col for col in df.columns if 'min' in col]
        
        # Process construct valid places
        input_places = pn[transition][0]
        accepted_places = []
        for place in input_places:
            for t in pn.keys():
                if place in pn[t][0] and transition != t:
                    other_input_places = pn[t][0]
                    other_input_places = [p for p in other_input_places if p not in input_places]
                    accepted_places.extend(other_input_places)
        
        outscope_places = [place for place in all_places if place not in accepted_places]
        outscope_cols = [col for col in df.columns if col.split('_')[0] in outscope_places]
        df = df.drop(token_count_cols + max_cols + min_cols + outscope_cols, axis=1)
        
        return df

    def _create_features_and_targets(self, df, transition):
        """Create feature matrix X and target vector y."""
        X = df.drop(['time', 'event_id'], axis=1)
        if 'consumed_token_id' in X.columns:
            X = X.drop('consumed_token_id', axis=1)
        
        y = df['event_id']
        y = y.apply(lambda x: f"{transition}_guard" if f"{transition}_constrained" in str(x) else x)

        # Convert all columns to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, y
    
    def _print_data_info(self, X):
        """Print information about the processed data."""
        print("\nData Info:")
        print(X.info())
        print("\nSample of data after cleaning:")
        print(X.head())

class DecisionTreeTrainer:
    """Class responsible for training decision tree models and visualizing them."""
    
    def __init__(self, max_depth=None, min_samples_split=2, test_size=0.2, random_state=42, 
                 figsize=(20, 10), dpi=300, save_visualizations=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.test_size = test_size
        self.random_state = random_state
        self.figsize = figsize
        self.dpi = dpi
        self.save_visualizations = save_visualizations
    
    def train_and_visualize_model(self, X, y, transition, pattern):
        """Train a decision tree model and optionally visualize it."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create and train the model
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Print performance metrics
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualize the tree if requested
        tree_filename = None
        if self.save_visualizations:
            tree_filename = self._visualize_tree(clf, X.columns, clf.classes_, transition, pattern)
        
        return clf, X_train, X_test, y_train, y_test, tree_filename
    
    def _visualize_tree(self, clf, feature_names, class_names, transition, pattern):
        """Create and save a visualization of the decision tree."""
        plt.figure(figsize=self.figsize)
        
        # Plot the tree
        tree.plot_tree(
            clf,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        # Save the tree
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'decision_tree_{transition}_{pattern}_{timestamp}.png'
        
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Decision tree visualization saved to: {filename}")
        return filename
    
    def save_test_predictions(self, y_test, y_pred, transition, pattern):
        """Save test predictions to Excel file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_transition = self._create_safe_filename(transition)
        
        test_predictions = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        
        filename = f'test_predictions_{safe_transition}_{pattern}_{timestamp}.xlsx'
        test_predictions.to_excel(filename, index=False)
        
        print(f"Test predictions saved to: {filename}")
        return filename
    
    def _create_safe_filename(self, transition):
        """Create a safe filename by replacing invalid characters."""
        return transition.translate(str.maketrans({
            ':': '_', ' ': '_', '-': '_', '<': '_', '>': '_', "'": '_'
        }))
    
class SimulationConfig:
    """Class to manage simulation configurations for different scenarios."""
    
    @staticmethod
    def get_config(file_path):
        """Get configuration based on file path."""
        if "priority" in file_path:
            return {
                'max_depth': 3,
                'patterns': ["priority"],
                'transitions': ["job_handling"],
                'pn': {
                    "job_arrival": (["arrival"], ["arrival", "q1"]),
                    "job_handling": (["q1", "r1"], ["r1"])
                }
            }
        
        elif "blocking" in file_path:
            return {
                'max_depth': 3,
                'patterns': ["blocking"],
                'transitions': ["pre_processing"],
                'pn': {
                    "pre_processing": (["arrival", "r1"], ["arrival", "r1", "q1"]),
                    "processing": (["q1", "r2"], ["r2"])
                }
            }
        
        elif "batching" in file_path:
            return {
                'max_depth': 2,
                'patterns': ["hold-batch"],
                'transitions': ["transportation"],
                'pn': {
                    "job_arrival": (["arrival"], ["arrival", "q1"]),
                    "transportation": (["q1", "r1"], ["r1"])
                }
            }
        
        elif "choice" in file_path:
            return {
                'max_depth': 5,
                'patterns': ["choice"],
                'transitions': ["game_production"],
                'pn': {
                    "chip_arrival": (["chip supply"], ["chip supply", "stock chip"]),
                    "phone_case_arrival": (["phone case supply"], ["phone case supply", "stock phone case"]),
                    "game_production": (["stock chip"], []),
                    "phone_production": (["stock chip", "stock phone case"], [])
                }
            }
        
        elif "SCM_game" or "scm_game" in file_path:
            # Get all event IDs ending with 'constrained' from the excel file
            df = pd.read_excel(file_path)
            transitions = [event_id.replace('_constrained', '') for event_id in df['event_id'].unique() if str(event_id).endswith('constrained')]
            
            return {
                'max_depth': 2,
                'patterns': ["priority", "blocking", "hold-batch", "choice"],
                'transitions': transitions,
                'pn': {
                    "order phone case<task:start>": (["source phone case"], ["source phone case", "stock phone cases"]),
                    "order chip<task:start>": (["source chip"], ["source chip", "stock chips"]),
                    "order game case<task:start>": (["source game case"], ["source game case", "stock game cases"]),
                    "prod phone": (["stock phone cases", "stock chips"], ["ffil phone NL"]),
                    "prod game": (["stock chips", "stock game cases"], ["ffil game NL"]),
                    "trans phone": (["ffil phone NL"], ["ffil phone DE"])
                }
            }
        
        else:
            raise ValueError(f"Unknown simulation type in file path: {file_path}")

class DecisionTreeAnalyzer:
    """Main class that orchestrates the entire decision tree analysis process."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = SimulationConfig.get_config(file_path) #TODO: generalize to use allow user input
        self.decision_sync_patterns = pd.DataFrame()

        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.trainer = DecisionTreeTrainer(
            max_depth=self.config['max_depth'],
            min_samples_split=5,
            save_visualizations=False
        )
    
    def run_analysis(self):
        """Run the complete decision tree analysis."""
        #print(f"Starting analysis for file: {self.file_path}")
        #print(f"Configuration: {self.config}")
        
        for transition in self.config['transitions']:
            for pattern in self.config['patterns']:
                print(f"\n{'='*50}")
                print(f"Transition: {transition}")
                print(f"Pattern: {pattern}")
                print(f"{'='*50}")
                
                try:
                    # Load and prepare data
                    X, y = self.preprocessor.load_and_prepare_data(
                        self.file_path, self.config['pn'], transition, pattern
                    )
                    
                    # Train model and visualize tree
                    safe_transition = self.trainer._create_safe_filename(transition)
                    clf, X_train, X_test, y_train, y_test, tree_filename = self.trainer.train_and_visualize_model(
                        X, y, safe_transition, pattern
                    )
                    
                    # Save test predictions
                    #predictions_filename = self.trainer.save_test_predictions(
                    #    y_test, clf.predict(X_test), safe_transition, pattern
                    #)
                    
                    # Run pattern analysis
                    pattern_analyzer = PatternDiscovery(
                        clf, X, self.config['pn'], transition, 
                        pattern_types=[pattern], 
                        leaf_samples_threshold=10, 
                        leaf_gini_threshold=0.1,
                        gini_decrease_threshold=0.01
                    )

                    # Analyze unconstrained patterns
                    pattern_analyzer.analyze_constrained_patterns(pattern)
                    
                    # Analyze constrained patterns
                    constrained_patterns = pattern_analyzer.analyze_constrained_patterns(pattern)
                    pattern_analyzer.print_constrained_pattern_summary(constrained_patterns, pattern)
                    self.decision_sync_patterns = pd.concat([self.decision_sync_patterns, constrained_patterns], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing transition {transition} with pattern {pattern}: {str(e)}")
                    continue
        
        self.decision_sync_patterns.to_excel(f'decision_sync_patterns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx', index=False)
        print(f"Decision sync patterns saved to: {f'decision_sync_patterns_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'}")


def main():
    """Main function to run the decision tree analysis."""
    experiment_nr = 10
    for i in range(1, experiment_nr+1):
    #    print(f"Running analysis {i} of {experiment_nr}")
        file_name = f"scm_game_log_{i}.xlsx"
        file_path = f"C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/{file_name}"
        analyzer = DecisionTreeAnalyzer(file_path)
        analyzer.run_analysis()
        print(f"Analysis {i} completed")
        print(f"{'='*50}")
    
    #file_name = "scm_game_exp2.xlsx"
    #file_path = f"C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/{file_name}"
    #file_path = "C:/Users/20183272/OneDrive - TU Eindhoven/Documents/PhD IS/Papers/Decision Synchronization Patterns/Modeling/Patterns/priority_report_20250819_164729.xlsx"
    #analyzer = DecisionTreeAnalyzer(file_path)
    #analyzer.run_analysis()

if __name__ == "__main__":
    main()