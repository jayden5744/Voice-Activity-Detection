import webrtcvad
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000


class Vis:
    @staticmethod
    def _norm_raw(raw):
        """
        Private function.
        Normalize the raw signal into a [0..1] range.
        """
        return raw / np.max(np.abs(raw), axis=0)

    @staticmethod
    def _time_axis(raw, labels):
        """
        Private function.
        Generates time axis for a raw signal and its labels.
        """

        time = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))
        time_labels = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(labels))
        return time, time_labels

    @staticmethod
    def _plot_waveform(frames, labels, title='Sample'):
        """
        Private function.
        Plot a raw signal as waveform and its corresponding labels.
        """
        raw = Vis._norm_raw(frames.flatten())
        time, time_labels = Vis._time_axis(raw, labels)

        plt.figure(1, figsize=(16, 3))
        plt.title(title)
        plt.plot(time, raw)
        plt.plot(time_labels, labels - 0.5)
        # plt.show()
        plt.savefig("./images/{}.jpg".format(title))
        plt.clf()

    @staticmethod
    def plot_sample(frames, labels, title='Sample', show_distribution=True):
        """
        Plot a sample with its original labels
        (before noise is applied to sample).
        """
        Vis._plot_waveform(frames, labels, title)

        # Print label distribution if enabled.
        if show_distribution:
            voice = (labels.tolist().count(1) * 100) / len(labels)
            silence = (labels.tolist().count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))

    @staticmethod
    def plot_sample_webrtc(frames, sensitivity=0):
        """
        Plot a sample labeled with WebRTC VAD
        (after noise is applied to sample).
        Sensitivity is an integer from 0 to 2,
        with 0 being the most sensitive.
        """
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames])
        Vis._plot_waveform(frames, labels, title='Sample (WebRTC)')

    @staticmethod
    def plot_features(mfcc=None, delta=None):
        """
        Plots the MFCC and delta-features
        for a given sample.
        """
        if mfcc is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(mfcc)
            plt.title('MFCC ({0} features)'.format(mfcc.shape[1]))
            plt.show()

        if delta is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(delta)
            plt.title('Deltas ({0} features)'.format(mfcc.shape[1]))
            plt.show()


class DataGenerator:
    def __init__(self, args, data, size_limit=0):
        self.args = args
        self.noise_level = None
        self.data = data
        self.size = size_limit if size_limit > 0 else len(data['labels'])
        self.data_mode = 0  # Default to training data

    def set_noise_level_db(self, level, reset_data_mode=True):
        noise_levels_db = {'None': None, '-15': -15, '-3': -3}
        if level not in noise_levels_db:
            raise Exception(f'Noise level "{level}" not supported! Options are: {list(noise_levels_db.keys())}')
        self.noise_level = level

        # Optionally reset data mode and position in file
        if reset_data_mode:
            if self.data_mode == 0:
                self.use_train_data()
            elif self.data_mode == 1:
                self.use_validate_data()
            elif self.data_mode == 2:
                self.use_test_data()

    def setup_generation(self, frame_count, step_size, batch_size, val_part=0.1, test_part=0.1):

        self.frame_count = frame_count
        self.step_size = step_size
        self.batch_size = batch_size

        # Setup indexes and sizes for data splits.
        self.train_index = 0
        self.val_index = int((1.0 - val_part - test_part) * self.size)
        self.test_index = int((1.0 - test_part) * self.size)

        self.train_size = self.val_index
        self.val_size = self.test_index - self.val_index
        self.test_size = self.size - self.test_index

    def use_train_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.train_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.train_index
        self.data_mode = 0

    def use_validate_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.val_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.val_index
        self.data_mode = 1

    def use_test_data(self):

        # Calculate how many batches we can construct from our given parameters.
        n = int((self.test_size - self.frame_count) / self.step_size) + 1
        self.batch_count = int(n / self.batch_size)
        self.initial_pos = self.test_index
        self.data_mode = 2

    def get_data(self, index_from, index_to):
        frames = self.data['frames-' + self.noise_level][index_from: index_to]
        mfcc = self.data['mfcc-' + self.noise_level][index_from: index_to]
        delta = self.data['delta-' + self.noise_level][index_from: index_to]
        labels = self.data['labels'][index_from: index_to]
        return frames, mfcc, delta, labels

    def get_batch(self, index):

        # Get current position.
        pos = self.initial_pos + (self.batch_size * index) * self.step_size

        # Get all data needed.
        l = self.frame_count + self.step_size * self.batch_size
        frames, mfcc, delta, labels = self.get_data(pos, pos + l)

        x, y, i = [], [], 0

        # Get batches
        while len(y) < self.batch_size:
            # Get data for the window.
            X = np.hstack((mfcc[i: i + self.frame_count], delta[i: i + self.frame_count]))

            # Append sequence to list of frames
            x.append(X)

            # Select label from center of sequence as label for that sequence.
            y_range = labels[i: i + self.frame_count]
            y.append(int(y_range[int(self.frame_count / 2)]))

            # Increment window using set step size
            i += self.step_size

        return x, y

    def plot_data(self, index_from, index_to, show_track=False):

        frames, mfcc, delta, labels = self.get_data(index_from, index_to)

        Vis.plot_sample(frames, labels)
        Vis.plot_sample_webrtc(frames)
        Vis.plot_features(mfcc, delta)
