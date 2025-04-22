import numpy as np

class SpatialPooler:
    def __init__(
        self,
        input_dim,
        column_dim,
        potential_pct=0.85,
        syn_perm_initial=0.21,
        syn_perm_active_inc=0.1,
        syn_perm_inactive_dec=0.01,
        connected_perm=0.5,
        stimulus_thresh=0,
        boost_strength=0.0,
        seed=None,
    ):
        self.input_dim = input_dim
        self.column_dim = column_dim
        self.potential_pct = potential_pct
        self.syn_perm_initial = syn_perm_initial
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.connected_perm = connected_perm
        self.stimulus_thresh = stimulus_thresh
        self.boost_strength = boost_strength
        self.seed = seed or np.random.randint(0, 1e6)
        self.rng = np.random.default_rng(self.seed)

        self._init_connections()

    def _init_connections(self):
        num_inputs = self.input_dim
        num_columns = self.column_dim

        # Determine potential pools (random subset of inputs for each column)
        self.potential_pools = [
            self.rng.choice(num_inputs, int(self.potential_pct * num_inputs), replace=False)
            for _ in range(num_columns)
        ]

        # Initialize synaptic permanences for potential synapses
        self.permanences = [
            self.rng.uniform(0, self.syn_perm_initial, size=len(pool))
            for pool in self.potential_pools
        ]

        # Binary mask of which synapses are "connected"
        self.connected_synapses = [
            perm >= self.connected_perm for perm in self.permanences
        ]

        # Boost factors (not yet adaptive)
        self.boost = np.ones(num_columns)

    def compute(self, input_vector):
        input_vector = np.array(input_vector)
        overlaps = np.zeros(self.column_dim)

        for col in range(self.column_dim):
            active_input_idxs = self.potential_pools[col][self.connected_synapses[col]]
            overlap = np.dot(input_vector[active_input_idxs], np.ones(len(active_input_idxs)))
            if overlap > self.stimulus_thresh:
                overlaps[col] = overlap * self.boost[col]

        # Winner-take-all inhibition (global for now)
        num_active = int(0.02 * self.column_dim)
        top_columns = overlaps.argsort()[::-1][:num_active]

        sp_output = np.zeros(self.column_dim, dtype=int)
        sp_output[top_columns] = 1

        return sp_output
