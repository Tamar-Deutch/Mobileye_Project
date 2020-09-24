class FrameContainer(object):
    def __init__(self, img_path):
        #self.img = plt.imread(img_path)
        self.img = img_path
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]

        self.colors = []
        self.cordinates = []
        self.cordinates_colors = []