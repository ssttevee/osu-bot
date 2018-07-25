from __future__ import print_function

import collections
import io
import math
import os
import tempfile
import types
import re
import zipfile

import audioread
import numpy as np
import resampy

from . import conv


Audio = collections.namedtuple("Audio", ["channels", "samplerate", "duration", "data"])


class _BeatMap:
    __name__ = "BeatMap"
    
    def __init__(self, info, audio_fn=None):
        self.__info = info
        self.__audio_fn = audio_fn
        self.__audio = None
    
    def __getattr__(self, attr):
        return self.__info[attr]
    
    def encoded_timing_points(self):
        audio = self.audio()
        millisecond_duration = audio.duration * 1000.
        
        parsed = [
            [float(n) for n in timing_point_str.split(",")[:2]] for timing_point_str in self.TimingPoints
        ]
        
        encoded_timing_points = []
        last_absolute_beat_duration = 0.
        for i in range(len(parsed)):
            if len(encoded_timing_points) < 1:
                prev_beat_duration = 0.
            else:
                _, _, prev_beat_duration = encoded_timing_points[-1]
            
            offset, beat_duration = parsed[i]
            
            if i == 0:
                offset = 0.
            
            if beat_duration < 0:
                if i == 0:
                    raise ValueError("first timing point is negative")
                beat_duration = last_absolute_beat_duration * (beat_duration / 100.) * -1
            else:
                last_absolute_beat_duration = beat_duration
            
            try:
                next_offset, _ = parsed[i + 1]
            except IndexError:
                next_offset = millisecond_duration
            
            if beat_duration == prev_beat_duration:
                encoded_timing_points[-1][1] = next_offset / millisecond_duration
                continue
            
            encoded_timing_points.append([
                offset / millisecond_duration,
                next_offset / millisecond_duration,
                beat_duration,
            ])
        
        # for timing_point in encoded_timing_points:
            # timing_point[2] = conv.beat_duration_to_bpm(timing_point[2])
        
        return np.array(encoded_timing_points)
        
    def audio(self):
        if self.__audio is None:
            self.__audio = self.__audio_fn()
        
        return self.__audio


def _parse_beatmap(reader, audio_fn=None):
    if not reader.readline().startswith('osu file format '):
        raise ValueError('Invalid osu beat map file')

    sections = {}
    for line in reader:
        line = line.rstrip()
        
        # skip commented lines
        if not line or line.lstrip()[:2] == '//':
            continue
    
        try:
            current_section = re.match(r'^\[\s*?(\w+)\s*?\]$', line).group(1)
        except AttributeError:
            try:
                try:
                    sections[current_section].append(line)
                except KeyError:
                    sections[current_section] = [line]
            except UnboundLocalError:
                # preamble, no section yet
                pass
    
    parsed_sections = {}
    for section, lines in sections.items():
        try:
            section_dict = {}
            for line in lines:
                matches = re.match(r'^\s*?(\w+)\s*?:\s*?(.*)$', line)
                key = matches.group(1)
                value = matches.group(2).strip()
                section_dict[key] = value
            parsed_sections[section] = collections.namedtuple(section, section_dict.keys())(*section_dict.values())
        except AttributeError:
            parsed_sections[section] = lines
    
    return _BeatMap(parsed_sections, audio_fn)


class BeatmapSet:
    __DEFAULT_BEATMAP_VERSION__ = b"default beat map"
    
    def __init__(self, file_or_buf):
        self.__buf = file_or_buf
        self.__zip = zipfile.ZipFile(file_or_buf)
        self.__maps = {}
        self.__audio = None
    
    def __del__(self):
        self.__buf.close()
    
    def __find_beatmap(self, version=None):
        for zipinfo in self.__zip.infolist():
            if not zipinfo.filename.endswith(".osu"):
                continue
            
            if version is not None and not zipinfo.filename.endswith("[%s].osu" % (version,)):
                continue
            
            return zipinfo
        
        return RuntimeError("Beatmap not found")
    
    def map(self, version=None, title=None, creator=None):
        try:
            return self.__maps[BeatmapSet.__DEFAULT_BEATMAP_VERSION__ if version is None else version]
        except KeyError:
            pass
        
        if title is not None and creator is not None:
            file = "%s (%s) [%s].osu" % (title, creator, version)
        else:
            file = self.__find_beatmap(version=version)
        
        with self.__zip.open(file, "r") as r:
            bm = _parse_beatmap(io.TextIOWrapper(r, "utf-8"), self.audio)
        
        if version is None:
            self.__maps[BeatmapSet.__DEFAULT_BEATMAP_VERSION__] = bm
            version = bm.Metadata.Version
        
        self.__maps[version] = bm
        
        return bm
    
    def audio(self, samplerate=48000, interval=0.01):
        if self.__audio is not None:
            return self.__audio
        
        filename = self.map().General.AudioFilename
        
        fh, path = tempfile.mkstemp()
        with open(fh, "wb") as w:
            with self.__zip.open(filename, "r") as r:
                w.write(r.read())
        
        with audioread.audio_open(path) as f:
            audio_samples = np.reshape(
                np.concatenate([
                    np.frombuffer(buf, dtype=np.int16) for buf in f
                ]),
                (f.channels, -1),
                'F',
            ).astype(np.float32)
        
        os.remove(path)
        
        # resample audio signal
        if f.samplerate != samplerate:
            audio_samples = resampy.resample(audio_samples, f.samplerate, samplerate, axis=1)
        
        # normalize samples
        normalized_samples = audio_samples / float(1 << 15)
        
        # pad samples to fit into frames
        samples_per_interval = int(interval * samplerate)
        sample_padding = samples_per_interval - (normalized_samples.shape[1] % samples_per_interval)
        padded_samples = np.pad(normalized_samples, ((0, 0), (0, sample_padding)), 'constant')
        
        # reshape samples into frames
        interval_frames = np.reshape(padded_samples, [f.channels, -1, int(interval * samplerate)])
        
        # apply fft
        interval_powers = np.abs(np.fft.fft(interval_frames))
        
        full_duration = audio_samples.shape[1] / f.samplerate
        self.__audio = Audio(f.channels, samplerate, full_duration, np.transpose(interval_powers, (1, 0, 2)))
        return self.__audio
