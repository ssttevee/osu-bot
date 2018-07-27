import glob
import random
import os
import sys
import time

import mxnet as mx

from nets.bpm_net import BPMNet, BPMDataLoader


checkpoints_dir = "checkpoints"
checkpoints_prefix = "bpm_net"


model_ctx = mx.gpu(0)


net = BPMNet()
net.hybridize()


epoch = 0
while True:
    params_path = os.path.join(checkpoints_dir, checkpoints_prefix + "-" + str(epoch) + ".params")
    if os.path.exists(params_path):
        latest_params_path = params_path
    else:
        break
    
    epoch += 1

try:
    net.load_params(latest_params_path, ctx=model_ctx)
except NameError:
    net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)


trainer = mx.gluon.Trainer(net.collect_params(), "adam")
L = mx.gluon.loss.L2Loss()


data_loader = BPMDataLoader("data", batch_size=5, shuffle=True)


try:
    while True:
        epoch += 1
        cumulative_loss = 0
        
        epoch_start_time = time.clock()
        for i, (audio, timing) in enumerate(data_loader):
            audio = audio.as_in_context(model_ctx)
            timing = timing.as_in_context(model_ctx)
            
            example_start_time = time.clock()
            with mx.autograd.record():
                output = net(audio, timing)
                
                # add sequence terminating element (i.e. all zeroes)
                y = mx.nd.concat(timing, mx.nd.zeros((1, *timing.shape[1:]), ctx=model_ctx), dim=0)
                
                loss = L(output, mx.nd.expand_dims(y, axis=1))
                example_loss = mx.nd.sum(loss).asscalar() / timing.shape[0]
            
            loss.backward()
            trainer.step(1)
            
            cumulative_loss += example_loss
            
            print("\rEpoch %d. Avg Loss: %s, Examples: %d, Avg Time/Batch: %.2fs, Last Batch Time: %.2fs" % (epoch, cumulative_loss/len(data_loader), i + 1, (time.clock() - epoch_start_time)/(i + 1), time.clock() - example_start_time), end="")
        
        os.makedirs(checkpoints_dir, exist_ok=True)
        net.save_params(os.path.join(checkpoints_dir, checkpoints_prefix + "-" + str(epoch) + ".params"))
        
        print()
except Exception as e:
    print()
    print()
    raise e
