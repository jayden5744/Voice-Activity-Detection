import time
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from utils import DataGenerator, Vis

obj_cuda = torch.cuda.is_available()


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.frames = args.frames
        self.batch_size = args.batch_size
        self.step_size = args.step_size
        self.features = args.features
        self.noise_levels = ['None', '-15', '-3']

    def test_predict(self, net, data, size_limit, noise_level):
        """
        Computes predictions on test data using given network.
        :param net:
        :param data:
        :param size_limit:
        :param noise_level:
        :return:
        """
        # Set up an instance of data generator using default partitions
        generator = DataGenerator(self.args, data, size_limit)
        generator.setup_generation(self.frames, self.step_size, self.batch_size)

        # Noise level does not match
        if noise_level not in self.noise_levels:
            raise Exception("Error: invalid noise level!")

        # When the training data connot be found
        if generator.test_size == 0:
            raise Exception("Error : no test data was found!")

        net.eval()
        generator.use_test_data()
        generator.set_noise_level_db(noise_level)

        y_true, y_score = [], []

        for i in range(generator.batch_count):
            x, y = generator.get_batch(i)
            x = Variable(torch.from_numpy(np.array(x)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            if obj_cuda:
                net = net.cuda()
                x = x.cuda()

            out = net(x)

            if obj_cuda:
                out = out.cpu()
                y = y.cpu()

            # Add true labels.
            y_true.extend(y.data.numpy())

            # Add probabilities for positive labels.
            y_score.extend(out.data.numpy()[:, 1])

        return y_true, y_score

    def roc_auc(self, nets, data, noise_lvl, size_limit=0):
        """
        Generates a ROC Curve for the given network and data for each noise level.
        :param nets:
        :param data:
        :param noise_lvl:
        :param size_limit:
        :return:
        """
        plt.figure(1, figsize=(16, 10))
        plt.title('Receiver Operating Characteristic (%s)' % noise_lvl, fontsize=16)

        # For each noise level
        for key in nets:
            net = nets[key]

            # Make Predictions
            y_ture, y_score = self.test_predict(net, data, size_limit, noise_lvl)

            # Compute ROC metrics and AUC
            fpr, tpr, thresholds = metrics.roc_curve(y_ture, y_score)
            auc_res = metrics.auc(fpr, tpr)

            # Plots the ROC Curve and show area
            plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (key, auc_res))


        plt.xlim([0, 0.2])
        plt.ylim([0.6, 1])
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.legend(loc='lower right', prop={'size': 16})
        plt.savefig("images/%s.jpg" % noise_lvl)
        plt.clf()
        # plt.show()

    def far(self, nets, data, size_limit=0, frr=1, plot=True):
        """
        Computes the confusion matrix for a given network.
        :param nets:
        :param data:
        :param size_limit:
        :param frr:
        :param plot:
        :return:
        """

        # Evaluate predictions using threshold
        def apply_threshold(y_score, t=0.5):
            return [1 if y >= t else 0 for idx, y in enumerate(y_score)]

        def fix_frr(y_ture, y_score, frr_target, noise_level):
            # Quick hack for initial threshold level to hit 1% FRR a bit faster
            if noise_level == 'None':
                t = 1e-4
            elif noise_level == '-15':
                t = 1e-5
            else:
                t = 1e-9

            # Compute FAR for a fixed FRR
            while t < 1.0:
                tn, fp, fn, tp = confusion_matrix(y_ture, apply_threshold(y_score, t)).ravel()

                far = (fp * 100) / (fp + tn)
                frr = (fn * 100) / (fn + tp)

                if frr >= frr_target:
                    return far, frr

                t *= 1.1

            # Return closest result if no good match found.
            return far, frr

        f = open("./images/far_frr.txt", 'w')

        for key in nets:
            net = nets[key]
            f.write(key + ':\n')
            f.write('Network metrics: \n')
            print(key + ':')
            print('Network metrics: ')
            for lvl in self.noise_levels:
                # Make predictions
                y_true, y_score = self.test_predict(net, data, size_limit, lvl)
                f.write(
                    'FAR: %0.2f%% for fixed FRR at %0.2f%% and noise level ' % fix_frr(y_true, y_score, frr, lvl)
                    + lvl + '\n')
                print('FAR: %0.2f%% for fixed FRR at %0.2f%% and noise level ' % fix_frr(y_true, y_score, frr, lvl),
                      lvl)
        print('-----Save FAR_FRR.txt-----')

    def netvad(self, net, data, noise_level='-3', init_pos=50, length=700, only_plot_net=False, timeit=True,
               net_name=''):
        """
        Generates a sample of specified length and runs it through the given network.
        By default, the network output is plotted alongside the original labels and WebRTC output for comparison
        """

        # Set up an instance of data generator using default partitions
        generator = DataGenerator(self.args, data)
        generator.setup_generation(self.frames, self.step_size, self.batch_size)

        # Noise level does not match
        if noise_level not in self.noise_levels:
            raise Exception("Error: invalid noise level!")

        # When the training data cannot be found
        if generator.test_size == 0:
            raise Exception("Error : no test data was found!")

        net.eval()
        generator.use_test_data()
        generator.set_noise_level_db(noise_level)

        raw_frames, mfcc, delta, labels = generator.get_data(init_pos, init_pos + length)

        # Convert sample to list of frames
        def get_frames():
            j = 0
            while j < length - self.frames:
                yield np.hstack((mfcc[j: j + self.frames], delta[j: j + self.frames]))
                j += 1

        # Start Timer
        if timeit:
            start_net = time.time()

        # Creates batches from frames
        frames = list(get_frames())
        batches, i, num_frames = [], 0, -1
        while i < len(frames):
            full = i + self.batch_size >= len(frames)
            end = i + self.batch_size if not full else len(frames)
            window = frames[i:end]
            if full:
                num_frames = len(window)
                while len(window) < self.batch_size:
                    window.append(np.zeros((self.frames, self.features)))
            batches.append(np.stack(window))
            i += self.batch_size

        # Predict for each frame
        offset = 15
        accum_out = [0] * offset
        for batch in batches:
            X = Variable(torch.from_numpy(batch).float())
            if obj_cuda:
                out = torch.max(net(X.cuda()), 1)[1].cpu().float().data.numpy()
            else:
                out = torch.max(net(X), 1)[1].float().data.numpy()
            accum_out.extend(out)

        # Stop timer
        if timeit:
            dur_net = str((time.time() - start_net) * 1000).split('.')[0]
            device = 'GPU' if obj_cuda else 'CPU'
            seq_dur = int((length / 100) * 3)
            print(f'Network processed {len(batches) * self.batch_size} frames ({seq_dur}s) in {dur_net}ms on {device}.')

        # Adjust padding
        if num_frames > 0:
            accum_out = accum_out[:len(accum_out) - (self.batch_size - num_frames)]
        accum_out = np.array(accum_out)

        # frames = np.array(frames)

        # Cut frames outside of prediction boundary
        raw_frames = raw_frames[offset:-offset]
        labels = labels[offset:-offset]
        accum_out = accum_out[offset:]

        # Plot results
        print('Displaying results for noise level:', noise_level)
        if not only_plot_net:
            Vis.plot_sample(raw_frames, labels, show_distribution=False)
            Vis.plot_sample_webrtc(raw_frames, sensitivity=0)
        Vis.plot_sample(raw_frames, accum_out, title=net_name, show_distribution=False)
