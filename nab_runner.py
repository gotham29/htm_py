import csv
import datetime
import logging
import numpy as np
from htm_py.htm_model import HTMModel
from htm_py.encoders.rdse import RDSE
from htm_py.encoders.date import DateEncoder
from htm_py.encoders.multi import MultiEncoder

logging.basicConfig(level=logging.INFO)

def load_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def main():
    # Parameters
    csv_path = "data/art_daily_jumpsup.csv"
    value_key = "value"
    timestamp_key = "timestamp"

    # Encoder setup
    enc_params = {
        "rdse": {"n": 150,
                 "w": 21,
                 "min_val": 0,
                 "max_val": 114.4,
                 "resolution": 0.88},  #for get a "resolution" 0.88 with numBuckets=130
        "date": {"timeOfDay": (21, 9.49)}
    }

    encoders = {
        "rdse": RDSE(min_val=enc_params['rdse']['min_val'],
                     max_val=114.4, n=150, w=21),  #resolution=0.1,  seed=42
        "date": DateEncoder(timeOfDay=enc_params['date']['timeOfDay'])   #weekend=(2, 1)
    }
    encoder = MultiEncoder(encoders)
    print(f"encoder.get_width() = {encoder.size}")


    sp_params = {
        "inputDimensions": (encoder.size,),
        "columnDimensions": (2048,),
        "potentialPct": 0.85,
        "globalInhibition": True,
        "numActiveColumnsPerInhArea": 40,
        "synPermConnected": 0.1,
        "stimulusThreshold": 0,
        "seed": 42
    }

    tm_params = {
        "columnDimensions": (2048,),
        "cellsPerColumn": 32,
        "activationThreshold": 13,
        "initialPermanence": 0.21,
        "connectedPermanence": 0.1,
        "minThreshold": 10,
        "maxNewSynapseCount": 20,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.1,
        "predictedSegmentDecrement": 0.0,
        "maxSegmentsPerCell": 128,
        "maxSynapsesPerSegment": 128,
        "seed": 42
    }

    model = HTMModel(encoder_params=enc_params, sp_params=sp_params, tm_params=tm_params)

    for i, row in enumerate(load_csv(csv_path)):
        # try:
        raw_value = float(row[value_key])
        dt = datetime.datetime.fromisoformat(row[timestamp_key])
        input_dict = {
            "rdse": raw_value,
            "date": dt
        }
        # print(f"time = {i}; input_dict = {input_dict}")
        print(f"time = {i}; input_dict = {input_dict}; encoded = {model.encoder.encode(input_dict)}")
        anomaly_score, prediction_count = model.compute(input_dict, learn=True, iteration=i)
        print(f"time={i} anomaly_score={anomaly_score:.3f} pred_count={prediction_count:.3f}")
        # except Exception as e:
        #     print(f"Error processing row {i}: {e}")

if __name__ == "__main__":
    main()
