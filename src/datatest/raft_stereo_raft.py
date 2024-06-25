RAFT_TORCHSCRIPTFILE = "models/raft_pointtrackSTIR.pt"
RAFT_STEREO_TORCHSCRIPTFILE = "models/iraftstereo_rvc_ontrackingpointsSTIR.pt"

import onnxruntime as ort
import torchvision
import cv2
import numpy as np
import torch
from testutil import backproject_2d_points
def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RAFTStereoRAFTTracker:
    def __init__(self):
        self.modeltype = "RAFTStereoRAFT"
        self.device = 'cuda'
        self.initialized = False
        self.queries = None
        self.scale = 2  # internal imsize is one-half of total
        self.internalimsize = (1280 // self.scale, 1024 // self.scale) # cv2 takes w, h
        self.tracker = torch.jit.load(RAFT_TORCHSCRIPTFILE).to(self.device)
        self.disparity_estimator = torch.jit.load(RAFT_STEREO_TORCHSCRIPTFILE).to(self.device)
        self.resize = torchvision.transforms.Resize((self.internalimsize[1], self.internalimsize[0]))

    def resize_and_convert_to_np(self, im):
        """ Takes a tensor, and resizes/converts it"""
        im = im.cpu().squeeze().numpy()
        im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
        return im

    from ipdb import iex
    @iex
    def trackpoints3D(self, pointlist, impair, impair_right, Q_pair, disparitypads_pair):
        """ Takes in RGB image"""
        with torch.no_grad():
            startim = self.resize_and_convert_to_np(impair[0]).transpose(2, 0, 1)/255.
            startim = impair[0].permute(0, 3, 1, 2) / 255.0
            endim = impair[1].permute(0, 3, 1, 2) / 255.0 # N C H W
            endim_right = impair_right[1].permute(0, 3, 1, 2) / 255.0

            startim = self.resize(startim.float().to(self.device))
            endim = self.resize(endim.float().to(self.device))
            endim_right = self.resize(endim_right.float().to(self.device))
            points = torch.from_numpy(pointlist[None]).to('cuda').float()

            points_scaled = points / self.scale
            end_points = self.tracker(points_scaled, startim, endim)
            disparities = self.disparity_estimator(points_scaled, endim, endim_right)

            estimated_endpoints = end_points * self.scale
            estimated_disparities = disparities * self.scale
            estimated_endpoints = to_numpy(estimated_endpoints.squeeze(0))
            estimated_disparities = to_numpy(estimated_disparities.squeeze(0))

            estimated_endpoints_3d = backproject_2d_points(estimated_endpoints, estimated_disparities, to_numpy(Q_pair[1][0]), to_numpy(disparitypads_pair[1]))

        return estimated_endpoints, estimated_endpoints_3d

