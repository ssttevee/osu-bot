import glob
import random
import os
import sys
import time

import numpy as np
import mxnet as mx
import h5py


checkpoints_dir = "checkpoints"
checkpoints_prefix = "bpm_net"


class BPMNet(mx.gluon.Block):
    def __init__(self, **kwargs):
        super(BPMNet, self).__init__(**kwargs)
        with self.name_scope():
            self.__encoder = mx.gluon.nn.Sequential()
            with self.__encoder.name_scope():
                self.__encoder.add(mx.gluon.rnn.LSTM(120))
                self.__encoder.add(mx.gluon.rnn.LSTM(120))
            
            self.__dense = mx.gluon.nn.Dense(3)
            
            self.__decoder = mx.gluon.nn.Sequential()
            with self.__decoder.name_scope():
                self.__decoder.add(mx.gluon.rnn.LSTM(3))
                self.__decoder.add(mx.gluon.rnn.LSTM(3))
                self.__decoder.add(mx.gluon.rnn.LSTM(3))

    '''
    forward expects parameters to be in TNC layout
    '''
    def forward(self, audio_fft_intervals, expected_output):
        """Forward pass
        
        Args:
            audio_fft_intervals: (required) pcm data of stero 48k samplerate
                audio reshaped to (t, n, 2, 480) and fft'd
            expected_output: (required) timing data in the shape of (t, n, 3)
                where the 3rd dim represents [relative_start_time, relative_end_time, beat_duration_ms]
        """
        
        # flatten everything past the 3rd axis
        x = mx.nd.reshape(audio_fft_intervals, shape=(0,0,-1))
        
        # reshape to TNC layout for encoder
        x = self.__encoder(x)
        
        # take the last element of the rnn output
        x = mx.nd.SequenceLast(x)
        
        # resize last rnn element to match decoder input size
        x = self.__dense(x)
        
        # add a time dimension to the front
        x = mx.nd.expand_dims(x, axis=0)
        
        # concat with the expected output on the time dimension
        x = mx.nd.concat(x, expected_output, dim=0)
        
        return self.__decoder(x)


model_ctx = mx.gpu(0)


os.makedirs(checkpoints_dir, exist_ok=True)


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

try:
    while True:
        epoch += 1
        cumulative_loss = 0
        
        data_files = glob.glob("data/*.ndarray")
        random.shuffle(data_files)
        
        start_time = time.clock()
        for i, data_file in enumerate(data_files):
            f = mx.nd.load(data_file)
            
            # audio data is in the shape of (n, 2, 480) where the dimensions represent (frames, channels, fft'd samples)
            audio = f["Audio"].as_in_context(model_ctx)
            
            # timing data is in the shape of (n, 3) where the 2nd dimension is [start_time, end_time, beat_duration_ms]
            timing = f["Timing"].as_in_context(model_ctx)
            
            with mx.autograd.record():
                output = net(
                    mx.nd.expand_dims(audio, axis=1),
                    mx.nd.expand_dims(timing, axis=1),
                )
                
                # add sequence terminating element (i.e. all zeroes)
                y = mx.nd.concat(timing, mx.nd.zeros((1, timing.shape[1]), ctx=model_ctx), dim=0)
                
                loss = L(output, mx.nd.expand_dims(y, axis=1))
                example_loss = mx.nd.sum(loss).asscalar() / timing.shape[0]
            
            loss.backward()
            trainer.step(1)
            
            cumulative_loss += example_loss
            
            print("\rEpoch %d. Cumulative Loss: %s, Examples: %d, Time/Example: %fs" % (epoch, cumulative_loss/len(data_files), i + 1, (time.clock() - start_time)/(i + 1)), end="")
        
        net.save_params(os.path.join(checkpoints_dir, checkpoints_prefix + "-" + str(epoch) + ".params"))
        
        print()
except Exception as e:
    print("failed on", data_file)
    raise e
