import glob
import h5py
import h5py_cache
import array
import webrtcvad
import numpy as np
from pydub import AudioSegment
import python_speech_features

np.random.seed(1337)


class FileManager:
    """
    Keeps track of audio-files from a data-set.
    Provides support for formatting the wav-files into a desired format.
    Also provides support for conversion of .flac files (as we have in the LibriSpeech data-set).
    """

    def __init__(self, args, name, directory):
        self.args = args
        self.name = name
        self.data = h5py.File(args.data_folder + '/' + name + '.hdf5', 'a')
        self.sample_rate = args.sample_rate
        self.sample_channels = args.sample_channels
        self.sample_width = args.sample_width
        self.frame_size_ms = args.frame_size_ms
        self.frame_size = int(self.sample_rate * (self.frame_size_ms / 1000.0))
        self.batch_size = 65536

        # Setup file names.
        if 'files' not in self.data:

            # Get files.
            files = glob.glob(directory + '/**/*.flac', recursive=True)
            files.extend(glob.glob(directory + '/**/**/**/**/*.flac', recursive=True))
            files = [f for f in files]

            # Setup data set.
            dt = h5py.special_dtype(vlen=str)
            self.data.create_dataset('files', (len(files),), dtype=dt)

            # Add file names.
            for i, f in enumerate(files):
                self.data['files'][i] = f

    def get_track_count(self):
        return len(self.data['files'])

    def prepare_files(self, normalize=False):
        '''
        Prepares the files for the project.
        Will do the following check for each file:
        1. Check if it has been converted already to the desired format.
        2. Converts all files to WAV with the desired properties.
        3. Stores the converted files in a separate folder.
        '''

        if not self.args.prepare_audio:
            print(f'Skipping check for {self.name}.')
            return

        print('Found {0} tracks to check.'.format(self.get_track_count()))
        progress = 1

        # Setup raw data set.
        if 'raw' not in self.data:
            dt = h5py.special_dtype(vlen=np.dtype(np.int16))
            self.data.create_dataset('raw', (self.get_track_count(),), dtype=dt)

        # Convert files to desired format and save raw content.
        for i, file in enumerate(self.data['files']):

            print('Processing {0} of {1}'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1

            # Already converted?
            if len(self.data['raw'][i]) > 0:
                continue

            # Convert file.
            track = (AudioSegment.from_file(file)
                     .set_frame_rate(self.sample_rate)
                     .set_sample_width(self.sample_width)
                     .set_channels(self.sample_channel))

            # Normalize?
            if normalize:
                track = track.apply_gain(-track.max_dBFS)

            # Store data.
            self.data['raw'][i] = np.array(track.get_array_of_samples(), dtype=np.int16)

        self.data.flush()
        print('\nDone!')

    def collect_frames(self):
        '''
        Takes all the audio files and merges their frames together into one long array
        for use with the sample generator.
        '''

        if 'frames' in self.data:
            print('Frame merging already done. Skipping.')
            return

        if 'raw' not in self.data:
            print('Could not find raw data!')
            return

        frame_count = 0
        progress = 1

        # Calculate number of frames needed.
        for raw in self.data['raw']:
            frame_count += int((len(raw) + (self.frame_size - (len(raw) % self.frame_size))) / self.frame_size)
            print('Counting frames ({0} of {1})'.format(progress, self.get_track_count()), end='\r', flush=True)
            progress += 1

        # Create data set for frames.
        dt = np.dtype(np.int16)
        self.data.create_dataset('frames', (frame_count, self.frame_size), dtype=dt)

        progress = 0

        # Buffer to speed up merging as HDF5 is not fast with lots of indexing.
        buffer = np.array([])
        buffer_limit = self.frame_size * 4096

        # Merge frames.
        for raw in self.data['raw']:

            # Setup raw data with zero padding on the end to fit frame size.
            raw = np.concatenate((raw, np.zeros(self.frame_size - (len(raw) % self.frame_size))))

            # Add to buffer.
            buffer = np.concatenate((buffer, raw))

            # If buffer is not filled up and we are not done, keep filling the buffer up.
            if len(buffer) < buffer_limit and progress + (len(buffer) / self.frame_size) < frame_count:
                continue

            # Get frames.
            frames = np.array(np.split(buffer, len(buffer) / self.frame_size))
            buffer = np.array([])

            # Add frames to list.
            self.data['frames'][progress: progress + len(frames)] = frames

            progress += len(frames)
            print('Merging frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('\nDone!')

    def label_frames(self):
        '''
        Takes all audio frames and labels them using the WebRTC VAD.
        '''

        if 'labels' in self.data:
            print('Frame labelling already done. Skipping.')
            return

        if 'frames' not in self.data:
            print('Could not find any frames!')
            return

        vad = webrtcvad.Vad(0)

        frame_count = len(self.data['frames'])
        progress = 0

        # Create data set for labels.
        dt = np.dtype(np.uint8)
        self.data.create_dataset('labels', (frame_count,), dtype=dt)

        # Label all the frames.
        for pos in range(0, frame_count, self.batch_size):
            frames = self.data['frames'][pos: pos + self.batch_size]
            labels = [1 if vad.is_speech(f.tobytes(), sample_rate=self.sample_rate) else 0 for f in frames]
            self.data['labels'][pos: pos + self.batch_size] = np.array(labels)

            progress += len(labels)
            print('Labelling frames ({0} of {1})'.format(progress, frame_count), end='\r', flush=True)

        self.data.flush()
        print('\nDone!')


class DataManager:
    """
    Now that all data is in the same format, we can construct the dataset for use in this project.
    Noise is added to speech in three different noise levels: none, low (-15 dB) and high (-3 dB).
    MFCCs and derivates are computed using a frame size of 30 ms and the entirety of the data is saved in a data.
    hdf5 file for later use. If data already has been processed, this step is skipped.
    """
    def __init__(self, args):
        self.args = args
        self.sample_rate = args.sample_rate
        self.sample_channels = args.sample_channels
        self.sample_width = args.sample_width
        self.frame_size_ms = args.frame_size_ms
        self.frame_size = int(self.sample_rate * (self.frame_size_ms / 1000.0))

    def prepare_data(self, speech_dataset, noise_dataset):
        data = h5py_cache.File(self.args.data_forlder + '/data.hdf5', 'a', chunk_cache_mem_size=1024 ** 3)
        noise_levels_db = {'None': None, '-15': -15, '-3': -3}
        mfcc_window_frame_size = 4
        slice_min = self.args.slice_min_ms
        slice_max = self.args.slice_max_ms

        speech_data = speech_dataset.data
        noise_data = noise_dataset.data

        if 'labels' not in data:

            print('Shuffling speech data and randomly adding 50% silence.')

            pos = 0
            l = len(speech_dataset.data['frames'])
            slices = []

            # Split speech data randomly within the given slice length.
            while pos + slice_min < l:
                slice_indexing = (pos, pos + np.random.randint(slice_min, slice_max + 1))
                slices.append(slice_indexing)
                pos = slice_indexing[1]

            # Add remainder to last slice.
            slices[-1] = (slices[-1][0], l)

            pos = 0

            # Add random silence (50%) to the track within the given slice length.
            while pos + slice_min < l:
                length = np.random.randint(slice_min, slice_max + 1)
                slice_indexing = (length, length)
                slices.append(slice_indexing)
                pos += length

            # Get total frame count.
            total = l + pos + mfcc_window_frame_size

            # Shuffle the content randomly.
            np.random.shuffle(slices)

            # Create data set for input.
            for key in noise_levels_db:
                data.create_dataset('frames-' + key, (total, self.frame_size), dtype=np.dtype(np.int16))
                data.create_dataset('mfcc-' + key, (total, 12), dtype=np.dtype(np.float32))
                data.create_dataset('delta-' + key, (total, 12), dtype=np.dtype(np.float32))

            # Create data set for labels.
            dt = np.dtype(np.int8)
            data.create_dataset('labels', (total,), dtype=dt)

            pos = 0

            # Construct speech data.
            for s in slices:

                # Silence?
                if s[0] == s[1]:
                    frames = np.zeros((s[0], self.frame_size))
                    labels = np.zeros(s[0])
                # Otherwise use speech data.
                else:
                    frames = speech_data['frames'][s[0]: s[1]]
                    labels = speech_data['labels'][s[0]: s[1]]

                # Pick random noise to add.
                i = np.random.randint(0, len(noise_data['frames']) - len(labels))
                noise = noise_data['frames'][i: i + len(labels)]

                # Setup noise levels.
                for key in noise_levels_db:

                    # Get previous frames to align MFCC window with new data.
                    if pos == 0:
                        align_frames = np.zeros((mfcc_window_frame_size - 1, self.frame_size))
                    else:
                        align_frames = data['frames-' + key][pos - mfcc_window_frame_size + 1: pos]

                    # Add noise and get frames, MFCC and delta of MFCC.
                    frames, mfcc, delta = self.add_noise(np.int16(frames), np.int16(noise),
                                                         np.int16(align_frames), noise_levels_db[key],
                                                         mfcc_window_frame_size)

                    data['frames-' + key][pos: pos + len(labels)] = frames
                    data['mfcc-' + key][pos: pos + len(labels)] = mfcc
                    data['delta-' + key][pos: pos + len(labels)] = delta

                # Add labels.
                data['labels'][pos: pos + len(labels)] = labels

                pos += len(labels)
                print('Generating data ({0:.2f} %)'.format((pos * 100) / total), end='\r', flush=True)

            data.flush()

            print('\nDone!')

        else:
            print('Speech data already generated. Skipping.')
        return data

    def add_noise(self, speech_frames, noise_frames, align_frames, noise_level_db, mfcc_window_frame_size=4):
        # Convert to tracks.
        speech_track = (AudioSegment(data=array.array('h', speech_frames.flatten()),
                                     sample_width=self.sample_width, frame_rate=self.sample_rate,
                                     channels=self.sample_channels))

        noise_track = (AudioSegment(data=array.array('h', noise_frames.flatten()),
                                    sample_width=self.sample_width, frame_rate=self.sample_rate,
                                    channels=self.sample_channels))

        # Overlay noise.
        track = noise_track.overlay(speech_track, gain_during_overlay=noise_level_db)

        # Get frames data from track.
        raw = np.array(track.get_array_of_samples(), dtype=np.int16)
        frames = np.array(np.split(raw, len(raw) / self.frame_size))

        # Add previous frames to align MFCC window.
        frames_aligned = np.concatenate((align_frames, frames))

        mfcc = python_speech_features.mfcc(frames_aligned, self.sample_rate, winstep=(self.frame_size_ms / 1000),
                                           winlen=mfcc_window_frame_size * (self.frame_size_ms / 1000), nfft=2048)

        # First MFCC feature is just the DC offset.
        mfcc = mfcc[:, 1:]
        delta = python_speech_features.delta(mfcc, 2)

        return frames, mfcc, delta
