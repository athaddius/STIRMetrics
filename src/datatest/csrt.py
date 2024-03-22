import cv2
import numpy as np
import torch

DEVICE = "cuda"


class CSRTMultiple:
    """ CSRT tracker. 
    Downscales input images by 2 to maintain performance"""
    def __init__(self):
        self.csrts = []
        self.firstframe = True
        self.modeltype = "CSRTMultiple"
        self.scale = 2  # internal imsize is one-half of total
        self.internalimsize = (1280 // self.scale, 1024 // self.scale)

    def initframetrackers(self, pointlist, im):
        im = im.cpu().squeeze().numpy()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
        h, w, c = im.shape
        for point in pointlist:
            tracker = cv2.TrackerCSRT_create()
            ret = tracker.init(im, CSRTMultiple.roifrompoint(point / self.scale, h, w))
            self.csrts.append(tracker)

    @staticmethod
    def roifrompoint(point, maxh, maxw):
        w = 29
        h = 29
        x = int(point[0] - w // 2)
        y = int(point[1] - h // 2)
        x = max(x, 0)
        y = max(y, 0)
        if x + w >= maxw:
            w = maxw - x
        if y + h >= maxh:
            h = maxh - y
        w = max(w, 1)
        h = max(h, 1)
        bbox = [x, y, w, h]
        return bbox

    def nextframe(self, im):
        positions = []
        im = im.cpu().squeeze().numpy()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, self.internalimsize)  ## scaled down by 2
        for tracker in self.csrts:
            ret, bbox = tracker.update(im)
            positions.append([bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2])

        positions = np.array(positions) * self.scale
        return positions

    def trackpoints2D(self, pointlist, impair):
        """ Returns: N, 2 pointlist in numpy, [xy]"""
        
        if self.firstframe:
            startim = impair[0]
            self.initframetrackers(pointlist, startim)
            self.firstframe = False
        endim = impair[1]
        pointlist = self.nextframe(endim)
        return pointlist


class CSRTMultiple3D:
    # Tracks in left primary, and uses right to backproject
    def __init__(self):
        self.modeltype = "CSRTMultiple3D"
        self.firstframe = True

    ##implement this
    def trackpoints(
        self,
        pointlist,
        pointlist_right,
        ims_ori_pair,
        ims_ori_pair_right,
        K,
        Q,
        disparitypad,
        startframe,
    ):
        if self.firstframe:
            self.lefttracker = CSRTMultiple()
            self.righttracker = CSRTMultiple()
            self.firstframe = False

        nextlocs = self.lefttracker.trackpoints2D(
            pointlist.cpu().squeeze(0).numpy(), ims_ori_pair
        )
        nextlocs_right = self.righttracker.trackpoints2D(
            pointlist_right.cpu().numpy().squeeze(0), ims_ori_pair_right
        )
        nextlocs = torch.tensor(nextlocs).to(DEVICE).unsqueeze(0)
        nextlocs_right = torch.tensor(nextlocs_right).to(DEVICE).unsqueeze(0)
        disparitypad = disparitypad.squeeze()
        disppoints = torch.stack(
            (
                nextlocs[:, :, 0],
                nextlocs[:, :, 1],
                (nextlocs[:, :, 0] + disparitypad) - nextlocs_right[:, :, 0],
            ),
            axis=-1,
        )  # npts 3
        disp_homogeneous = torch.nn.functional.pad(
            disppoints, ((0, 1)), mode="constant", value=1
        )  # B N 3
        disp_homogeneous = torch.bmm(disp_homogeneous, Q.permute(0, 2, 1))
        disp_xyz = disp_homogeneous[:, :, :3] / disp_homogeneous[:, :, 3:4]
        return nextlocs, nextlocs_right, disp_xyz
