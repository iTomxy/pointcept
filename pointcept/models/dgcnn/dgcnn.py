import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p, bfs_cluster = None, None

from pointcept.models.losses import build_criteria
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.builder import MODELS

__all__ = ["DGCNN_partseg", "DGCNN_semseg"]


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


@MODELS.register_module()
class DGCNN(nn.Module):
    """(30 Sept 2025, iTom) used as backbone
    Adapted from DGCNN_semseg, so far cannot adjust architecture by args.
    """
    def __init__(self, in_channels=3, k=20, emb_dims=1024, out_channels=256):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, out_channels, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=dropout)
        # self.conv9 = nn.Conv1d(out_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, input_dict):
        """
        Input:
            input_dict:
              `- coord: [bs, in_channels, #points]
        Output:
            x: [bs, out_channels, #points]
        """
        x = input_dict["coord"]
        bs = x.size(0)
        npoint = x.size(2)

        # (bs, 9, npoint) -> (bs, 9*2, npoint, k)
        x = get_graph_feature(x, k=self.k, dim9=True)
        # (bs, 9*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv1(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv2(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x1, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv3(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv4(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x2, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv5(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)      # (bs, 64*3, npoint)

        # (bs, 64*3, npoint) -> (bs, emb_dims, npoint)
        x = self.conv6(x)
        # (bs, emb_dims, npoint) -> (bs, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, npoint)          # (bs, 1024, npoint)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (bs, 1024+64*3, npoint)

        # (bs, 1024+64*3, npoint) -> (bs, 512, npoint)
        x = self.conv7(x)
        # (bs, 512, npoint) -> (bs, out_channels, npoint)
        x = self.conv8(x)
        # x = self.dp1(x)
        # (bs, out_channels, npoint) -> (bs, 13, npoint)
        # x = self.conv9(x)
        # (bs, 13, npoint) -> (bs, npoint, 13)
        # x = x.transpose(2, 1).contiguous()

        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40, n_additional_channels=0):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6+2*n_additional_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)

        return x


class Transform_Net(nn.Module):
    # def __init__(self, args, in_channels=3):
    def __init__(self, in_channels=3):
        """
        iTom's modifications:
            - in_channels: int = 3, input dimension, default 3 means xyz
        """
        super(Transform_Net, self).__init__()
        # self.args = args
        self.k = 3
        self.in_channels = in_channels

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, in_channels * in_channels)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(in_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, inc*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, inc*inc)
        x = x.view(batch_size, self.in_channels, self.in_channels)          # (batch_size, inc*inc) -> (batch_size, inc, inc)

        return x


@MODELS.register_module()
class DGCNN_partseg(nn.Module):
    # def __init__(self, args, seg_num_all, num_categories=16, in_channels=3):
    def __init__(self, seg_num_all, num_categories, in_channels=3, k=40, emb_dims=1024, dropout=0.5):
        """
        seg_num_all: int, num of all part classes from all object classes
        num_categories: int, num of object classes (not part classes)
        in_channels: int = 3, input dimension, default 3 means xyz
        k: int = 40, Num of nearest neighbors to use
        emb_dims: int = 1024
        dropout: float = 0.5, dropout rate
        """
        super(DGCNN_partseg, self).__init__()
        # self.args = args
        self.seg_num_all = seg_num_all
        self.k = k
        self.transform_net = Transform_Net(in_channels)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(num_categories, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    # def forward(self, x, l):
    def forward(self, input_dict):
        """
        Input:
            input_dict:
              |- coord: [bs, in_channels, #points]
              `- cls_token: [bs, num_categories]
        Output:
            pred: [bs, seg_num_all, #points]
        """
        x = input_dict["coord"]
        l = input_dict["cls_token"]
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        pred = self.conv11(x)                   # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return pred


# @MODELS.register_module()
# class DGCNN_semseg(nn.Module):
#     def __init__(self, num_classes, npoints, in_channels=3, k=20, emb_dims=1024, out_emb_dims=256, dropout=0.5,
#         criteria=[
#             dict(type="FocalLoss", loss_weight=1.0),
#             dict(type="DiceLoss", loss_weight=1.0)
#         ]
#     ):
#         super(DGCNN_semseg, self).__init__()
#         self.k = k
#         self.npoints = npoints
#         self.in_channels = in_channels
#         self.criteria = build_criteria(criteria)

#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.bn6 = nn.BatchNorm1d(emb_dims)
#         self.bn7 = nn.BatchNorm1d(512)
#         self.bn8 = nn.BatchNorm1d(out_emb_dims)

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(2 * in_channels, 64, kernel_size=1, bias=False),
#             self.bn1,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=1, bias=False),
#             self.bn2,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#             self.bn3,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=1, bias=False),
#             self.bn4,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#             self.bn5,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(
#             nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
#             self.bn6,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = nn.Sequential(
#             nn.Conv1d(1216, 512, kernel_size=1, bias=False),
#             self.bn7,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.conv8 = nn.Sequential(
#             nn.Conv1d(512, out_emb_dims, kernel_size=1, bias=False),
#             self.bn8,
#             nn.LeakyReLU(negative_slope=0.2))
#         self.dp1 = nn.Dropout(p=dropout)
#         self.conv9 = nn.Conv1d(out_emb_dims, num_classes, kernel_size=1, bias=False)

#     def forward(self, input_dict):
#         """
#         Input:
#             input_dict:
#               `- coord: [bs, in_channels, #points]
#         Output:
#             x: [bs, num_classes, #points]
#         """
#         x = input_dict["coord"]
#         print(x.size())
#         x = x.view(-1, self.npoints, self.in_channels) # (bs*npt, in_channels) -> (bs, npt, in_channels)
#         x = x.permute(0, 2, 1) # -> (bs, in_channels, #points)
#         print("after:", x.size())
#         bs = x.size(0)
#         npoint = x.size(2)

#         # (bs, 9, npoint) -> (bs, 9*2, npoint, k)
#         x = get_graph_feature(x, k=self.k, dim9=True)
#         # (bs, 9*2, npoint, k) -> (bs, 64, npoint, k)
#         x = self.conv1(x)
#         # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
#         x = self.conv2(x)
#         # (bs, 64, npoint, k) -> (bs, 64, npoint)
#         x1 = x.max(dim=-1, keepdim=False)[0]

#         # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
#         x = get_graph_feature(x1, k=self.k)
#         # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
#         x = self.conv3(x)
#         # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
#         x = self.conv4(x)
#         # (bs, 64, npoint, k) -> (bs, 64, npoint)
#         x2 = x.max(dim=-1, keepdim=False)[0]

#         # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
#         x = get_graph_feature(x2, k=self.k)
#         # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
#         x = self.conv5(x)
#         # (bs, 64, npoint, k) -> (bs, 64, npoint)
#         x3 = x.max(dim=-1, keepdim=False)[0]

#         x = torch.cat((x1, x2, x3), dim=1)      # (bs, 64*3, npoint)

#         # (bs, 64*3, npoint) -> (bs, emb_dims, npoint)
#         x = self.conv6(x)
#         # (bs, emb_dims, npoint) -> (bs, emb_dims, 1)
#         x = x.max(dim=-1, keepdim=True)[0]

#         x = x.repeat(1, 1, npoint)          # (bs, 1024, npoint)
#         x = torch.cat((x, x1, x2, x3), dim=1)   # (bs, 1024+64*3, npoint)

#         # (bs, 1024+64*3, npoint) -> (bs, 512, npoint)
#         x = self.conv7(x)
#         # (bs, 512, npoint) -> (bs, out_emb_dims, npoint)
#         x = self.conv8(x)
#         # x = self.dp1(x)
#         # (bs, out_emb_dims, npoint) -> (bs, 13, npoint)
#         x = self.conv9(x)
#         # (bs, 13, npoint) -> (bs, npoint, 13)
#         x = x.transpose(2, 1).contiguous()

#         # return x
#         logit = x
#         # logit = x.transpose(2, 1).contiguous() # -> (bs, npoint, num_classes)
#         # print("logit:", logit.size())
#         logit = logit.view(-1, logit.size(-1)) # -> (bs*npoint, num_classes)
#         print("logit:", logit.size())
#         loss = self.criteria(logit, input_dict["segment"])
#         if self.training:
#             return dict(loss=loss)
#         else:
#             # logit_ = logit.permute(0, 2, 1).contiguous() # [bs, npt, num_classes]
#             # logit_ = logit_.view(-1, logit_.shape[-1]) # [bs*npt, num_classes]
#             return dict(loss=loss, seg_logits=logit)


@MODELS.register_module()
class DGCNN_semseg(nn.Module):
    def __init__(self, num_classes, npoints,
        in_channels=3, k=20, emb_dims=1024, feat_dims=256, dropout=0.5,
        criteria=[
            dict(type="FocalLoss", loss_weight=1.0),
            dict(type="DiceLoss", loss_weight=1.0)
        ]
    ):
        super(DGCNN_semseg, self).__init__()
        self.criteria = build_criteria(criteria)
        self.npoints = npoints
        self.in_channels = in_channels
        self.backbone = DGCNN(in_channels, k, emb_dims, feat_dims)
        self.cls_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(feat_dims, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, input_dict):
        """
        Input:
            input_dict:
              `- coord: [bs*npt, in_channels]
        """
        x = input_dict["coord"]
        x = x.view(-1, self.npoints, self.in_channels) # (bs*npt, in_channels) -> (bs, npt, in_channels)
        x = x.transpose(2, 1) # -> (bs, in_channels, #points)
        feat = self.backbone({"coord": x}) # (bs, feat_dims, npt)
        logit = self.cls_head(feat) # (bs, num_classes, npt)
        logit = logit.transpose(2, 1).contiguous() # -> [bs, npt, num_classes]
        logit = logit.view(-1, logit.shape[-1]) # -> [bs*npt, num_classes]
        loss = self.criteria(logit, input_dict["segment"])
        if self.training:
            return dict(loss=loss)
        else:
            return dict(loss=loss, seg_logits=logit)


@MODELS.register_module()
class DGCNN_clreg(nn.Module):
    """Centre Line REGression
    Input FG points, use oracle segment in grouping.
    """

    def __init__(self, npoints,
        in_channels=3, k=20, emb_dims=1024, feat_dims=256,
        semantic_num_classes=1+1,
        instance_ignore_index=-1,
        # log_dist=False, # log the distance (like R-CNN)
        # voxel_size=0.02,
        segment_ignore_index=(-1, 0),
        cluster_thresh=10.0,
        cluster_closed_points=100,
        cluster_propose_points=100,
        cluster_min_points=50,
    ):
        super(DGCNN_clreg, self).__init__()
        self.semantic_num_classes = semantic_num_classes
        self.npoints = npoints
        self.in_channels = in_channels
        self.instance_ignore_index = instance_ignore_index
        # self.voxel_size = voxel_size
        self.segment_ignore_index = segment_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.backbone = DGCNN(in_channels, k, emb_dims, feat_dims)
        self.bias_head = nn.Sequential( # from PointGroup
            nn.Linear(feat_dims, feat_dims),
            nn.BatchNorm1d(feat_dims, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(feat_dims, 3),
        )

    def forward(self, input_dict, npoints=None):
        """
        Input:
            input_dict:
              `- coord: [bs*npt, in_channels]
            npoints: int = None, specify #points of each sample in the batch
        """
        x = input_dict["coord"]
        npoints = self.npoints if npoints is None else npoints
        x = x.view(-1, npoints, self.in_channels) # (bs*npt, in_channels) -> (bs, npt, in_channels)
        x = x.transpose(2, 1) # -> (bs, in_channels, #points)
        feat = self.backbone({"coord": x}) # (bs, feat_dims, npt)
        feat = feat.transpose(2, 1).contiguous() # -> [bs, npt, feat_dims]
        feat = feat.view(-1, feat.size(-1)) # -> [bs*npt, feat_dims]
        bias_pred = self.bias_head(feat) # (bs*npt, 3)

        coord = input_dict["coord"]
        instance_centroid = input_dict["instance_centroid"] # [bs*npt, 3]
        instance = input_dict["instance"] # [bs*npt]
        offset = input_dict["offset"] # [bs], [npt, 2*npt, ..., bs*npt]

        mask = (instance != self.instance_ignore_index).float()
        bias_gt = instance_centroid - coord
        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
            torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (
            torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
        )
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
            torch.sum(mask) + 1e-8
        )

        loss = bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            loss=loss,
            bias_l1_loss=bias_l1_loss,
            bias_cosine_loss=bias_cosine_loss,
        )
        if not self.training:
            return_dict["bias_pred"] = bias_pred # [bs*npt, 3]

            center_pred = coord + bias_pred
            # center_pred /= self.voxel_size
            oracle_logit_pred = torch.zeros_like(coord)[:, :2]
            oracle_logit_pred[:, 1] = 1
            seg_logits = logit_pred = oracle_logit_pred # only for testing when seg branch is not introduced
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            # cluster
            mask = ( # points predicted to be not in `segment_ignore_index`
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            # mask.sum(): n_fg_pt, #{points not ignored}, i.e. foreground points
            if mask.sum() == 0:
                # all points are to be ignored
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                # offset2batch(offset): [bs*npt], [[0]*npt, ..., [bs-1]*npt]
                batch_ = offset2batch(offset)[mask]
                # batch2offset(batch_): [0, n1, n1+n2, ..., bs*npt]
                # nn.ConstantPad1d: prepend a `0` to the list
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                # idx: [n_all_neighbor_pt], neighbor point indices of each fg point
                # start_len: [n_fg_pt, 2], start index and length of each fg point's neighbors in `idx`
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                # proposals_idx: [n_fg_pt, 2], which point belongs to which cluster
                #   - proposals_idx[:, 0]: cluster id
                #   - proposals_idx[:, 1]: point index
                # proposals_offset: [n_cluster+1], cluster start and end index in proposals_idx
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                # map (FG) point index back to original (ALL) indices
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # get proposal
            # proposals_pred: int[n_cluster, npt], in {0, 1}, cluster assignment matrix
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            # instance_pred: int[n_cluster], in {0, ..., semantic_num_classes-1}, object class for each cluster
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            proposals_point_num = proposals_pred.sum(1) # [n_cluster], #points of each cluster
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
            return_dict["seg_logits"] = seg_logits

        return return_dict
