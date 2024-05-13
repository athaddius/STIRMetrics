ONNXFILE = "models/raft_pointtrackSTIR.onnx"
TORCHSCRIPTFILE = "models/raft_pointtrackSTIR.pt"
import onnxruntime as ort
import torchvision
import cv2
import numpy as np
import torch
def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RAFTTracker:
    def __init__(self):
        self.modeltype = "RAFT"
        self.use_torchscript = True
        self.device = 'cuda'
        self.initialized = False
        self.queries = None
        self.scale = 2  # internal imsize is one-half of total
        self.internalimsize = (1280 // self.scale, 1024 // self.scale) # cv2 takes w, h
        if self.use_torchscript:
            self.tracker = torch.jit.load(TORCHSCRIPTFILE).to(self.device)
            self.resize = torchvision.transforms.Resize((self.internalimsize[1], self.internalimsize[0]))
        else:
            self.tracker = ort.InferenceSession(ONNXFILE)

    def resize_and_convert_to_np(self, im):
        """ Takes a tensor, and resizes/converts it"""
        im = im.cpu().squeeze().numpy()
        im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
        return im

    from ipdb import iex
    @iex
    def trackpoints2D(self, pointlist, impair):
        """ Takes in RGB image"""

        if self.use_torchscript:
            startim = self.resize_and_convert_to_np(impair[0]).transpose(2, 0, 1)/255.
            startim = impair[0].permute(0, 3, 1, 2) / 255.0
            endim = impair[1].permute(0, 3, 1, 2) / 255.0 # N C H W
            startim = self.resize(startim.float().to(self.device))
            endim = self.resize(endim.float().to(self.device))
            points = torch.from_numpy(pointlist[None]).to('cuda').float()


            points_scaled = points / self.scale
            end_points = self.tracker(points_scaled, startim, endim)
            estimated_endpoints = end_points * self.scale
            estimated_endpoints = to_numpy(estimated_endpoints.squeeze(0))
        else:
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


