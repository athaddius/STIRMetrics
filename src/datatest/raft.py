ONNXFILE = "models/raft_pointtrackSTIR.onnx"
import onnxruntime as ort
import cv2
import numpy as np
import torch
def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RAFTTracker:
    def __init__(self):
        self.modeltype = "RAFT"
        self.tracker = ort.InferenceSession(ONNXFILE)
        self.initialized = False
        self.queries = None
        self.scale = 2  # internal imsize is one-half of total
        self.internalimsize = (1280 // self.scale, 1024 // self.scale)

    def resize_and_convert_to_np(self, im):
        im = im.cpu().squeeze().numpy()
        im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
        return im

    from ipdb import iex
    @iex
    def trackpoints2D(self, pointlist, impair):
        """ Takes in RGB image"""

        startim = self.resize_and_convert_to_np(impair[0]).transpose(2, 0, 1)/255.
        endim = self.resize_and_convert_to_np(impair[1]).transpose(2, 0, 1)/255.
        startim = startim[None,...].astype(np.float32)
        endim = endim[None,...].astype(np.float32)
        points = pointlist[None,...].astype(np.float32)


        points_scaled = points / self.scale
        outputs = self.tracker.run(
                    None,
                    {"pointlist": points_scaled, "image1": startim, "image2": endim},
                        )
        estimated_endpoints = outputs[0] * self.scale
        estimated_endpoints = estimated_endpoints.squeeze(0)
        return estimated_endpoints


