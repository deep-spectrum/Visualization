# Specify the recordings to segment. They should all be the same 'type'.
import json
import os

import numpy as np
import sigmf
from sigmf import SigMFFile
recordings_directory = "/media/daniel-mayer/Deep-Spectrum/"
recordings_to_segment = [
    {"name": "NonVibrating-RX-Recording-PureTone-1KHz-Cosine-2.45GHz-1",
     "output_segment_base_name": "/home/daniel-mayer/Documents/Segmented Recordings/NonVibrating-RX-PureTone-2.45GHz-Segmented",
     "desired_segments": [[0.1, 60.1], [60.1, 120.1], [120.1, 180.1], [180.1, 240.1], [240.1, 300.1], [300.1, 360.1],
                          [420.1, 480.1], [480.1, 540.1], [540.1, 600.1], [600.1, 660.1], [660.1, 720.1],
                          [720.1, 780.1]],
     }
]
processed_recordings = []





memory_mapped_recordings = []

def to_np_datatype(raw_datatype):
    if raw_datatype == "cf32_le":
        return np.complex64
    # ... add more as needed ...

# Obtain the datatypes and sampling rates.
for recording_to_segment in recordings_to_segment:
    recording_name = recording_to_segment["name"]
    # Read in the datatype and sampling rate.
    with open(recordings_directory + recording_name + ".sigmf-meta", "r") as f:
        json_data = json.load(f)
        # Read in the raw values.
        raw_datatype = json_data["global"]["core:datatype"]
        raw_sampling_rate = json_data["global"]["core:sample_rate"]

        # Convert them to the correct formats.
        datatype_np = to_np_datatype(raw_datatype)
        sampling_rate = float(raw_sampling_rate)

        # Create the object for the recording.
        processed_recordings.append({
            "datatype_np": datatype_np,
            "sampling_rate": sampling_rate,
            "name": recording_name,
            "desired_segments": recording_to_segment["desired_segments"],
            "output_segment_base_name": recording_to_segment["output_segment_base_name"],
            "original_meta": json_data
        })

# Determine the number of available samples in each recording.
for recording in processed_recordings:
    # Get the file size.
    file_size = os.path.getsize(recordings_directory + recording["name"] + '.sigmf-data')

    # Determine how many samples.
    num_samples = int(file_size / np.dtype(datatype_np).itemsize)

    print("Num samples: ", num_samples, " as time ", num_samples / sampling_rate, " seconds")

    recording["num_samples"] = num_samples


# Check that the requested segments are possible.
for recording in processed_recordings:
    recording["desired_segments_in_samples"] = []
    for desired_segment in recording["desired_segments"]:
        desired_segment_start_in_samples = desired_segment[0] * sampling_rate
        desired_segment_end_in_samples = desired_segment[1] * sampling_rate

        if desired_segment_end_in_samples >= recording["num_samples"] or desired_segment_start_in_samples >= recording["num_samples"]:
            desired_segment_end_in_samples = min(recording["num_samples"] - 1, desired_segment_end_in_samples)
            desired_segment_start_in_samples = min(recording["num_samples"] - 1, desired_segment_start_in_samples)

            print("WARNING: Requested segment exceeds bounds of the full data file.")
        recording["desired_segments_in_samples"].append((desired_segment_start_in_samples, desired_segment_end_in_samples))


# Use memory-mapping to copy out the desired number of samples in each recording.
for recording in processed_recordings:
    print("Processing recording: ", recording["name"])
    data_filename = recordings_directory + recording["name"] + ".sigmf-data"
    data_mmap = np.memmap(data_filename, dtype=recording["datatype_np"], mode="r")

    for index in range(0, len(recording["desired_segments_in_samples"])):
        print("Processing segment ", index)
        # Write the segment data.
        segment = data_mmap[int(recording["desired_segments_in_samples"][index][0]) : int(recording["desired_segments_in_samples"][index][1])]
        segment_filename = recording["output_segment_base_name"] + "Seg" + str(index) + ".sigmf-data"
        output_memmap = np.memmap(segment_filename, dtype=recording["datatype_np"], mode="w+", shape=segment.shape)
        output_memmap[...] = segment

        # Write the segment metadata.
        original_meta = recording["original_meta"]
        original_meta["global"]["seg_start"] = recording["desired_segments"][index][0]
        original_meta["global"]["seg_end"] = recording["desired_segments"][index][1]
        original_meta["global"]["seg_note"] = "This recording represents a segment of a longer recording, bounded in seconds by seg_start and seg_end."
        with open(recording["output_segment_base_name"] + "Seg" + str(index) + ".sigmf-meta", "w") as f:
            json.dump(original_meta, f)

        del output_memmap
        print("Done with segment", index)
