CONFIG = "MFT_files/configs/MFT_cfg.py"
from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import cv2
import torch

class MFTTracker:
    def __init__(self):
        config = load_config(CONFIG)
        self.modeltype = "MFT"
        self.tracker = config.tracker_class(config)
        self.initialized = False
        self.queries = None
        self.scale = 2  # internal imsize is one-half of total
        self.internalimsize = (1280 // self.scale, 1024 // self.scale)

    from ipdb import iex
    @iex
    def trackpoints2D(self, pointlist, impair):
        """ Takes in RGB image"""
        ## expects input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
        def convert_to_np_bgr(im):
            im = im.cpu().squeeze().numpy()
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
            return im

        if not self.initialized:
            self.queries = torch.from_numpy(pointlist).float().cuda() ## check format
            self.queries = self.queries / self.scale
            meta = self.tracker.init(convert_to_np_bgr(impair[0]))
            self.initialized = True
        meta = self.tracker.track(convert_to_np_bgr(impair[1])) # check frame format
        coords, occlusions = convert_to_point_tracking(meta.result, self.queries)
        #coords = # N, x, y, height width size
        return coords * self.scale


