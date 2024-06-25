import numpy as np
import torch
import cv2
from scipy.spatial import KDTree
import csrt
import mft
import raft

device = "cuda"

modeldict = {"MFT": mft.MFTTracker,
           "CSRT": csrt.CSRTMultiple,
           "RAFT": raft.RAFTTracker}

def todevice(cpudict):
    outdict = {}
    for k, v in cpudict.items():
        if k != "ims_ori":
            outdict[k] = [x.to(device) for x in v]
        else:
            outdict[k] = v
    return outdict


def showimage(name, im):
    """ Shows image using opencv at half resolution"""
    h, w = im.shape[:2]
    im = cv2.resize(im, (w // 2, h // 2))
    cv2.imshow(name, im)


def iou(ptsa, ptsb, h, w):
    """chamfer loss between point sets
    N 2"""
    im1 = np.zeros((h, w), dtype=np.int)
    im2 = np.zeros((h, w), dtype=np.int)
    if type(ptsa) == torch.Tensor:
        ptsa = ptsa.cpu().numpy()
    if type(ptsb) == torch.Tensor:
        ptsb = ptsb.cpu().numpy()
    for pt in ptsa:
        x = round(pt[0])
        y = round(pt[1])
        im1[x, y] = 1
    for pt in ptsb:
        x = round(pt[0])
        y = round(pt[1])
        im2[x, y] = 1
    intersection = np.bitwise_and(im1, im2).sum()
    union = np.bitwise_or(im1, im2).sum()
    iou = intersection / union
    return iou


def pointlossunidirectional(ptsa, ptsb):
    """point loss between ptsa, and nearest in ptsb
    N 2"""
    if type(ptsa) == torch.Tensor:
        ptsa = ptsa.cpu().numpy()
    if type(ptsb) == torch.Tensor:
        ptsb = ptsb.cpu().numpy()
    num_point = ptsa.shape[0]
    tree2 = KDTree(ptsb, leafsize=10)
    distances2, indices = tree2.query(ptsa)  # find closest to ptsa
    av_dist2 = np.mean(distances2)  # average euclidean distance
    pointdists = distances2

    if ptsa.shape[0] == 1:
        indices = [indices]
    for idx in indices:
        if idx == ptsb.shape[0]:
            breakpoint()
    displacements = [ptsb[ind] - ptsa[i] for i, ind in enumerate(indices)]

    return {
        "averagedistance": av_dist2,
        "distancelist": pointdists.tolist(),
        "displacements": displacements,
    }


def positionsfromim(segim):  # x,y positions
    """takes in segmentation image and returns x,y locations in halfscale of segim res"""
    inds = np.nonzero(segim)  # tuple of inds
    positions = np.array([list(reversed(s)) for s in zip(*inds)])  # positions n,2
    positions = positions
    return positions.astype(np.float32)


# https://stackoverflow.com/questions/47060685/chamfer-distance-between-two-point-clouds-in-tensorflow
def chamferloss(ptsa, ptsb):
    """chamfer loss between point sets
    N 2"""
    ptsa = ptsa.cpu().numpy()
    ptsb = ptsb.cpu().numpy()
    num_point = ptsa.shape[0]
    tree1 = KDTree(ptsa, leafsize=10)
    tree2 = KDTree(ptsb, leafsize=10)
    distances1, _ = tree1.query(ptsb)  # find closest to ptsb
    distances2, _ = tree2.query(ptsa)  # find closest to ptsa
    av_dist1 = np.mean(distances1)
    av_dist2 = np.mean(distances2)
    dist = av_dist1 + av_dist2
    return dist

def backproject_2d_points(points2d, disparities, Q, disparitypad):
    """Backprojects 2d points to 3d; taken from STIRLoader.
    Args:
        points2d: npts 2
        disparities: npts
        Q: 4x4
        disparitypad: scalar
    Returns:
        points3d: npts 3

    """
    padded_disparities = disparities + disparitypad

    disppoints = np.stack(
        (points2d[:, 0], points2d[:, 1], padded_disparities[:, 0]), axis=-1
    )  # npts 3
    disp_homogeneous = np.pad(
        disppoints, ((0, 0), (0, 1)), "constant", constant_values=1
    )
    disp_homogeneous = disp_homogeneous @ Q.T
    points3d = disp_homogeneous[:, :3] / disp_homogeneous[:, 3:4]
    return points3d
