import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

from frame_container import FrameContainer
from tfl_attention import find_tfl_lights
from confirm_tfl_use_CNN import confirm_tfl_by_CNN
import sfm as SFM


#========================Auxiliary Function===========================

def get_model():
    model=load_model("first.h5")
    return model

def visualize_dots(part, title, candidates, colors, img):
    part.set_title(title)
    part.imshow(plt.imread(img))

    candidates = np.array(candidates)
    red = colors.count('red')
    part.plot(candidates[red:, 0], candidates[red:, 1], 'ro', color='g', markersize=2)
    part.plot(candidates[:red, 0], candidates[:red, 1], 'ro', color='r', markersize=2)

def visualize_distances(part, traffic_light, traffic_lights_3d_location, valid):
    for i in range(len(traffic_light)):
        if valid[i]:
            part.text(traffic_light[i, 0], traffic_light[i, 1],
                       r'{0:.1f}'.format(traffic_lights_3d_location[i, 2]), color='y')

#========================End of Auxiliary Functions===========================

class TFL_Manager(object):

    def __init__(self, pp, focal):
        self.principle_point = pp
        self.focal_length = focal
        self.prev_frame = None
        self.curr_frame = None
        self.model = get_model()
        self.id_frame = 0

    def run(self, frame, EM):
        self.curr_frame = FrameContainer(frame)
        self.curr_frame.EM = EM

        #part 1
        self.curr_frame.cordinates, self.curr_frame.cordinates_colors = find_tfl_lights(self.curr_frame.img)

        #part 2
        confirm_tfl_by_CNN(self.curr_frame, self.model)

        # part 3
        if self.prev_frame:
            SFM.calc_TFL_dist(self.prev_frame, self.curr_frame, self.focal_length, self.principle_point)
            # run_sfm(self.curr_frame, self.prev_frame, self.principle_point, self.focal_length)


    def visualize(self):

        fig, (part1, part2, part3) = plt.subplots(1, 3,figsize=(12,6))
        fig.suptitle(f'Frame #{self.id_frame}  {self.curr_frame.img}')

        #part 1
        visualize_dots(part1, "candidates", self.curr_frame.cordinates, self.curr_frame.cordinates_colors, self.curr_frame.img)

        #part 2
        visualize_dots(part2, "traffic_lights", self.curr_frame.traffic_light, self.curr_frame.colors,self.curr_frame.img)
        print(self.curr_frame.traffic_light)
        #part 3
        if self.prev_frame:
            visualize_dots(part3, "distance", self.curr_frame.traffic_light, self.curr_frame.colors,self.curr_frame.img)
            visualize_distances(part3, self.curr_frame.traffic_light, self.curr_frame.traffic_lights_3d_location, self.curr_frame.valid)

        plt.show(block=True)

    def end(self):
        self.prev_frame = self.curr_frame
        self.id_frame += 1