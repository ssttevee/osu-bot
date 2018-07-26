import os

import mxnet

from osu.api import OsuAPIClient

client = OsuAPIClient("ssttevee", "37mem!9$FOO*0j25Wt9r@nn$")

data_path = "data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

for beatmap_result in client.search(limit=1000):
    output_path = os.path.join(data_path, "%d.ndarray" % beatmap_result["id"])
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
    
    if a.channels != 2:
        print(beatmap_result["id"], "skipping... unexpected number of channels:", a.channels)
        continue
    
    print(beatmap_result["id"], a.data.shape, timing_data.shape)
    
    mxnet.nd.save(output_path, {
        "Audio": mxnet.nd.array(a.data),
        "Timing": mxnet.nd.array(timing_data),
    })
