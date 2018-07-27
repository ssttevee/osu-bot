import functools
import glob

import mxnet as mx


class BPMNet(mx.gluon.Block):
    def __init__(self, encoder_depth=2, encoder_width=128, 
                 intermediate_depth=1, decoder_depth=3, **kwargs):
        super(BPMNet, self).__init__(**kwargs)
        with self.name_scope():
            self.__encoder = mx.gluon.nn.Sequential()
            with self.__encoder.name_scope():
                for _ in range(encoder_depth):
                    self.__encoder.add(mx.gluon.rnn.LSTM(encoder_width))
            
            self.__intermediate = mx.gluon.nn.HybridSequential()
            with self.__intermediate.name_scope():
                for _ in range(intermediate_depth):
                    self.__intermediate.add(mx.gluon.nn.Dense(3))
            
            self.__decoder = mx.gluon.nn.Sequential()
            with self.__decoder.name_scope():
                for _ in range(decoder_depth):
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
        x = self.__intermediate(x)
        
        # add a time dimension to the front
        x = mx.nd.expand_dims(x, axis=0)
        
        # concat with the expected output on the time dimension
        x = mx.nd.concat(x, expected_output, dim=0)
        
        return self.__decoder(x)


class BPMDataset(mx.gluon.data.Dataset):
    def __init__(self, data_dir, keys=("Audio", "Timing")):
        self.__file_paths = glob.glob(data_dir + "/*.ndarray")
        self.__data_keys = keys
    
    def __len__(self):
        return len(self.__file_paths)
        
    def __getitem__(self, i):
        f = mx.nd.load(self.__file_paths[i])
        return tuple([f[key] for key in self.__data_keys])


def max_dim_len(*data, axis=0):
    return functools.reduce(lambda a, b: a if a.shape[axis] > b.shape[axis] else b, data).shape[axis]


def bpm_data_batchify_fn(data):
    if isinstance(data[0], tuple):
        return tuple([bpm_data_batchify_fn(i) for i in zip(*data)])
    
    t = max_dim_len(*data)
    def pad_array(a):
        diff = t - a.shape[0]
        if diff <= 0:
            return a
        return mx.nd.concat(a, mx.nd.zeros((diff, *a.shape[1:])), dim=0)
    
    return mx.nd.stack(*map(pad_array, data), axis=1)


class BPMDataLoader(mx.gluon.data.DataLoader):
    def __init__(self, data_dir, **kwargs):
        super(BPMDataLoader, self).__init__(BPMDataset(data_dir), batchify_fn=bpm_data_batchify_fn, **kwargs)
