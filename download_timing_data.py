import os

import h5py
import numpy

from osu.api import OsuAPIClient

client = OsuAPIClient("ssttevee", "37mem!9$FOO*0j25Wt9r@nn$")

data_path = "h5data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

for beatmap_result in client.search(limit=1000):
    output_path = os.path.join(data_path, "%d.h5" % beatmap_result["id"])
    if os.path.exists(output_path):
        continue
    
    try:
        beatmap_set = client.download_beatmap(beatmap_result["id"])
    except Exception as e:
        print(beatmap_result["id"], "failed to download beatmap set:", str(e))
        continue
    
    try:
        a = beatmap_set.audio()
        timing_data = beatmap_set.map().encoded_timing_points().astype('float32')
    except Exception as e:
        print(beatmap_result["id"], "failed to prepare data:", str(e))
        continue
    
    print(beatmap_result["id"], a.data.shape, timing_data.shape)
    
    with h5py.File(output_path) as f:
        f.create_dataset("Audio", data=a.data)
        f.create_dataset("Timing", data=timing_data)
    
    # a.data.tofile(os.path.join(path, "audio.dat"))
    # timing_data.tofile(os.path.join(path, "timing.dat"))
