import numpy as np
from htm_py.connections import Connections
import os

tm_trace_path = "results/tm_phase_trace.csv"
if not os.path.exists(tm_trace_path):
    with open(tm_trace_path, "w") as f:
        f.write("timestep,phase,event,cell,segment,info\n")


class TemporalMemory:
    def __init__(self, column_dimensions, cells_per_column, activation_threshold,
                 initial_permanence, connected_permanence, min_threshold,
                 max_new_synapse_count, permanence_increment, permanence_decrement,
                 predicted_segment_decrement, seed=None, max_segments_per_cell=255,
                 max_synapses_per_segment=255, check_inputs=False):
        self.column_dimensions = column_dimensions
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.min_threshold = min_threshold
        self.max_new_synapse_count = max_new_synapse_count
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.predicted_segment_decrement = predicted_segment_decrement
        self.check_inputs = check_inputs

        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment

        self.seed = seed if seed is not None else np.random.randint(0, 100000)
        np.random.seed(self.seed)

        # Model state
        self.active_cells = set()
        self.winner_cells = set()
        self.active_segments = set()
        self.matching_segments = set()
        
        self.connections = Connections()

        # Additional state for learning
        self.iteration = 0
        self.last_used_iteration_for_segment = {}


    def compute(self, active_columns, learn=True):
        self.activate_dendrites(learn)
        self.activate_cells(active_columns, learn)

        # Compute Anomaly Score before advancing time
        anomaly = self.anomaly_score(active_columns)

        num_active_columns = len(active_columns)
        num_predictive_cells = len(self.get_predictive_cells())

        # === Phase 3: Prediction Accuracy Logging BEFORE advancing iteration ===
        predicted_cells = self.get_predictive_cells()
        predicted_columns = set(
            self.connections.column_for_cell(cell, self.cells_per_column) 
            for cell in predicted_cells
        )

        active_columns_set = set(active_columns)
        correct_predictions = predicted_columns.intersection(active_columns_set)
        wrong_predictions = predicted_columns - active_columns_set

        log_path = "results/tm_phase3_prediction_accuracy_detailed.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("timestep,predicted_column,is_correct\n")

        with open(log_path, "a") as f:
            for col in predicted_columns:
                is_correct = int(col in correct_predictions)
                f.write(f"{self.iteration},{col},{is_correct}\n")

        prediction_count = (num_predictive_cells / num_active_columns) if num_active_columns > 0 else 0.0

        # Phase 3: Advance time AFTER logging
        self.iteration += 1

        return anomaly, prediction_count


    def activate_dendrites(self, learn=True):
        """
        Phase 1: Compute active and matching segments based on the current active cells.

        Args:
            learn (bool): If True, learning updates will be applied.
        """

        with open(tm_trace_path, "a") as f:
            for segment in self.active_segments:
                cell = self.connections.cell_for_segment(segment)
                f.write(f"{self.iteration},Phase1,SegmentActive,{cell},{segment},active_connected_synapses\n")
            for segment in self.matching_segments:
                cell = self.connections.cell_for_segment(segment)
                f.write(f"{self.iteration},Phase1,SegmentMatching,{cell},{segment},active_potential_synapses\n")

        self.active_segments = set()
        self.matching_segments = set()

        # Precompute segment activations
        for segment in self.connections.segments():
            active_connected_synapses = self.connections.num_active_connected_synapses(
                segment, self.active_cells, self.connected_permanence
            )

            if active_connected_synapses >= self.activation_threshold:
                self.active_segments.add(segment)

            active_potential_synapses = self.connections.num_active_potential_synapses(
                segment, self.active_cells
            )

            if active_potential_synapses >= self.min_threshold:
                self.matching_segments.add(segment)

        if learn:
            print(f"Iteration {self.iteration}: matching_segments = {self.matching_segments}")
            print(f"Iteration {self.iteration}: active_segments = {self.active_segments}")
            for segment in self.matching_segments:
                if segment not in self.active_segments:
                    self.connections.adapt_segment(
                        segment,
                        prev_active_cells=self.active_cells,
                        permanence_increment=0.0,
                        permanence_decrement=self.predicted_segment_decrement,
                        iteration=self.iteration
                    )
                    with open("results/tm_phase_trace.csv", "a") as f:
                        f.write(f"{self.iteration},Phase1,PredictedSegmentDecrementApplied,,{segment},failed_prediction\n")


    def activate_cells(self, active_columns, learn=True):
        """
        Phase 2: Activate cells based on predictive state or burst if necessary.
        """
        prev_active_cells = self.active_cells.copy()
        prev_winner_cells = self.winner_cells.copy()

        self.active_cells.clear()
        self.winner_cells.clear()

        for column in active_columns:
            predictive_cells = [
                cell for cell in self.cells_for_column(column)
                if self.connections.is_cell_predictive(cell, self.active_segments)
            ]

            if predictive_cells:
                # Predicted Column Activation
                for cell in predictive_cells:
                    self.active_cells.add(cell)
                    self.winner_cells.add(cell)
                    if learn:
                        segments = self.connections.segments_for_cell(cell)
                        for segment in segments:
                            if segment in self.active_segments:
                                self.connections.adapt_segment(
                                    segment, prev_active_cells,
                                    self.permanence_increment, self.permanence_decrement, self.iteration
                                )
                                with open(tm_trace_path, "a") as f:
                                    f.write(f"{self.iteration},Phase2,AdaptSegment,{cell},{segment},predicted\n")
            else:
                # Bursting Column - No predictions available
                for cell in self.cells_for_column(column):
                    self.active_cells.add(cell)

                winner_cell = self.select_winner_cell(column)
                self.winner_cells.add(winner_cell)

                with open(tm_trace_path, "a") as f:
                    f.write(f"{self.iteration},Phase2,BurstWinnerCell,{winner_cell},,burst\n")

                if learn:
                    matching_segments = self.connections.matching_segments_for_column(
                        column, self.cells_per_column, prev_active_cells, self.min_threshold
                    )

                    if matching_segments:
                        best_segment = max(
                            matching_segments,
                            key=lambda s: self.connections.num_active_potential_synapses(s, prev_active_cells)
                        )
                        self.connections.adapt_segment(
                            best_segment, prev_active_cells,
                            self.permanence_increment, self.permanence_decrement, self.iteration
                        )
                        with open(tm_trace_path, "a") as f:
                            f.write(f"{self.iteration},Phase2,AdaptSegment,{winner_cell},{best_segment},burst_matched\n")
                    else:
                        # Always grow a new segment if no matching segment found!
                        new_segment = self.connections.create_segment(winner_cell)
                        self.connections.grow_synapses(
                            new_segment, prev_winner_cells,
                            self.initial_permanence, self.max_new_synapse_count
                        )
                        with open(tm_trace_path, "a") as f:
                            f.write(f"{self.iteration},Phase2,SegmentGrown,{winner_cell},{new_segment},new_segment_burst\n")

        segment_log_path = "results/tm_segment_growth_trace.csv"
        if not os.path.exists(segment_log_path):
            with open(segment_log_path, "w") as f:
                f.write("timestep,total_segments,total_synapses,avg_permanence\n")

        # Collect total segments and synapse permanence data
        total_segments = len(self.connections.segment_to_synapses)
        all_permanences = [
            self.connections.synapse_data[synapse_id][1]
            for synapse_ids in self.connections.segment_to_synapses.values()
            for synapse_id in synapse_ids
        ]
        total_synapses = len(all_permanences)
        avg_permanence = np.mean(all_permanences) if total_synapses > 0 else 0.0

        with open(segment_log_path, "a") as f:
            f.write(f"{self.iteration},{total_segments},{total_synapses},{avg_permanence:.4f}\n")


    def select_winner_cell(self, column):
        """
        Selects the cell with the fewest segments in the given column.
        If multiple candidates exist, choose randomly.

        Args:
            column (int): Column index.

        Returns:
            int: Winner cell index.
        """
        start_cell = column * self.cells_per_column
        end_cell = start_cell + self.cells_per_column

        min_segments = float('inf')
        candidate_cells = []

        for cell in range(start_cell, end_cell):
            num_segments = self.connections.num_segments(cell)
            if num_segments < min_segments:
                min_segments = num_segments
                candidate_cells = [cell]
            elif num_segments == min_segments:
                candidate_cells.append(cell)

        return np.random.choice(candidate_cells)


    def get_predictive_cells(self):
        """
        Returns the set of cells that are in a predictive state for the next timestep.

        Returns:
            set of int: Indices of predictive cells.
        """
        with open("results/tm_phase3_prediction_trace.csv", "a") as f:
            if self.iteration == 0:  # Write header only once
                f.write("timestep,predictive_cell,column\n")

            for segment in self.active_segments:
                cell = self.connections.cell_for_segment(segment)
                column = self.connections.column_for_cell(cell, self.cells_per_column)
                f.write(f"{self.iteration},{cell},{column}\n")

        predictive_cells = set()

        for segment in self.active_segments:
            cell = self.connections.cell_for_segment(segment)
            predictive_cells.add(cell)

        return predictive_cells


    def cells_for_column(self, column):
        """
        Returns a list of cell indices for the given column.

        Args:
            column (int): Column index.

        Returns:
            list of int: Cell indices.
        """
        start_cell = column * self.cells_per_column
        end_cell = start_cell + self.cells_per_column
        return list(range(start_cell, end_cell))


    def create_segment(self, cell):
        """
        Creates a new segment on the given cell. Handles segment limits.

        Args:
            cell (int): The cell index to create the segment on.

        Returns:
            Segment object or ID from Connections.
        """
        return self.connections.create_segment(cell)


    def anomaly_score(self, active_columns):
        """
        Calculates the anomaly score based on how many active columns 
        were NOT predicted.

        Args:
            active_columns (list of int): Indices of active columns at this timestep.

        Returns:
            float: Anomaly score between 0.0 and 1.0
        """
        num_active_columns = len(active_columns)
        if num_active_columns == 0:
            return 0.0  # No activity, no anomaly.

        predictive_columns = set(
            self.connections.column_for_cell(cell, self.cells_per_column)
            for cell in self.get_predictive_cells()
        )

        num_predicted_columns = sum(1 for col in active_columns if col in predictive_columns)

        return 1.0 - (num_predicted_columns / num_active_columns)


