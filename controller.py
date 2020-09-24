import pickle
import numpy as np
from tfl_manager import TFL_Manager


class Controller(object):

    def __init__(self, clip):
        with open(f"{clip}.pls", "r") as pls_file:

            lines = pls_file.read().splitlines()
            self.frame_list = lines[1:]
            pkl_file_path = lines[0]

            with open(pkl_file_path, 'rb') as pkl_file:
                data = pickle.load(pkl_file, encoding='latin1')
                pp = data['principle_point']
                focal = data['flx']

                self.EMs = [None]
                i = int(self.frame_list[0].split('_')[2][-2:])
                for i in range(i, i +len(self.frame_list) - 1):
                    EM = np.eye(4)
                    EM = np.dot(data[f'egomotion_{i}-{i + 1}'], EM)
                    self.EMs.append(EM)

        self.TFL_Man = TFL_Manager(pp, focal)

    def run(self):
        for i, frame in enumerate(self.frame_list):
            self.TFL_Man.run(frame, self.EMs[i])
            self.TFL_Man.visualize()
            self.TFL_Man.end()


def main():
    my_controller = Controller('examples/example')
    my_controller.run()

if __name__ == '__main__':
    main()