import numpy as np
import pytest
from htm_py.spatial_pooler import SpatialPooler


class TestSPAdvanced:

    def setup_method(self):
        self.sp = SpatialPooler(
            inputDimensions=(8,),
            columnDimensions=(4,),
            potentialPct=1.0,
            globalInhibition=True,
            numActiveColumnsPerInhArea=2,
            stimulusThreshold=0,
            synPermConnected=0.2,
            synPermActiveInc=0.1,
            synPermInactiveDec=0.05,
            boostStrength=1.0,
            seed=123
        )
        self.input_all_ones = np.ones(8, dtype=np.int8)
        self.input_all_zeros = np.zeros(8, dtype=np.int8)

    def test_boosting_increases_over_time(self):
        original_boosts = self.sp._boostFactors.copy()
        for _ in range(10):
            output = np.zeros(self.sp.numColumns, dtype=np.int8)
            self.sp.compute(self.input_all_ones, learn=True, output=output)
        boosted = self.sp._boostFactors
        assert np.any(boosted > original_boosts)

    def test_duty_cycles_update(self):
        for _ in range(5):
            output = np.zeros(self.sp.numColumns, dtype=np.int8)
            self.sp.compute(self.input_all_ones, learn=True, output=output)
        assert np.all(self.sp._activeDutyCycles >= 0)
        assert np.all(self.sp._overlapDutyCycles >= 0)

    def test_permanence_clipping(self):
        col_idx = 0
        for _ in range(5):
            self.sp._adapt_permanences(col_idx, self.input_all_ones)
        assert np.all(self.sp._permanences[col_idx] <= 1.0)
        assert np.all(self.sp._permanences[col_idx] >= 0.0)

        for _ in range(5):
            self.sp._adapt_permanences(col_idx, self.input_all_zeros)
        assert np.all(self.sp._permanences[col_idx] <= 1.0)
        assert np.all(self.sp._permanences[col_idx] >= 0.0)

    def test_potential_pool_mask_respected(self):
        self.sp.potentialPct = 0.5
        self.sp._potential_inputs = np.zeros((self.sp.numColumns, self.sp.numInputs), dtype=bool)
        self.sp._potential_inputs[0, :4] = True  # Only allow first half
        self.sp._permanences[0, :4] = 0.2
        self.sp._permanences[0, 4:] = 0.8  # Should not be affected

        self.sp._adapt_permanences(0, self.input_all_ones)

        # These should be unchanged because outside potential pool
        assert np.all(self.sp._permanences[0, 4:] == 0.8)

    def test_deterministic_behavior_with_seed(self):
        sp1 = SpatialPooler(inputDimensions=(8,), columnDimensions=(4,), seed=42)
        sp2 = SpatialPooler(inputDimensions=(8,), columnDimensions=(4,), seed=42)

        np.testing.assert_array_almost_equal(sp1._permanences, sp2._permanences)
        np.testing.assert_array_equal(sp1.connectedSynapses, sp2.connectedSynapses)

    def test_sparse_representation_size(self):
        for _ in range(5):
            output = np.zeros(self.sp.numColumns, dtype=np.int8)
            self.sp.compute(self.input_all_ones, learn=True, output=output)
            num_active = np.sum(output)
            assert num_active <= self.sp.numActiveColumnsPerInhArea
            assert num_active >= 0