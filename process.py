import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MPIIDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_name)
        keypoints = self.annotations.iloc[idx, 1:-3].values.astype('float32').reshape(-1, 2)
        scale = float(self.annotations.iloc[idx, -3])  # Person scale w.r.t. 200 px height
        img_height, img_width = image.shape[:2]
        orig_height = (img_height / scale) * 200
        orig_width = (img_width / scale) * 200

        # sample = {'image': image, 'keypoints': keypoints, 'orig_height': orig_height, 'orig_width': orig_width}
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        image = cv2.resize(image, (new_w, new_h))
        keypoints = keypoints * [new_w / w, new_h / h]

        return {'image': image, 'keypoints': keypoints}

class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # numpy image: H x W x C to torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'image': transforms.ToTensor()(image), 'keypoints': torch.tensor(keypoints)}


class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) #Gaussian Kernel

    def __call__(self, joints):
        hms = np.zeros((joints.shape[0], self.num_joints, self.output_res, self.output_res), dtype=np.float64)
        sigma = self.sigma
        # print(f"Joint's shape : {joints.shape}")
        for i, p in enumerate(joints):
            # print(p.shape)
            for idx, pt in enumerate(p):
                if(pt[0] > 0):
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[i, idx, aa:bb, cc:dd] = np.maximum(hms[i, idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

def _unravel_index_2D(index, shape):
    # tnp does not contain np.unravel_index. This is a 2D version of the latter
    return np.array([index//shape[1] % shape[0], index % shape[1]])
    

class HeatmapToKeypointsConverter:
    def __init__(self):
        pass

    def __call__(self, heatmap_grid):
        # Reshape the heatmap grid to easily find the index of the maximum 'value'
        # print("---")
        # print(heatmap_grid.shape)
        # flattened_heatmaps = heatmap_grid.view(heatmap_grid.size(0), heatmap_grid.size(1), heatmap_grid.size(2), -1)
        # print(flattened_heatmaps.shape)
        # max_indices = torch.argmax(flattened_heatmaps, dim=3)

        # # Calculate x, y coordinates from the flattened indices
        # y_coords = max_indices // heatmap_grid.size(2)
        # x_coords = max_indices % heatmap_grid.size(2)
        # print(x_coords.shape)

        # # Stack x, y coordinates to form keypoints tensor
        # keypoints = torch.stack((x_coords, y_coords), dim=)

        # return keypoints

        ################

        # N, H, W, C = heatmap_grid.shape

        # # Transpose to channels first
        # heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))

        # # Reshape each heatmap to a single dimension e.g. 3x3 to 9
        # # Combine batches of heatmaps to one larger list of heatmaps
        # # Then take argmax and retreive 2D keypoint through unraveling
        # keypoint_preds = _unravel_index_2D(
        #     np.argmax(np.reshape(heatmaps, (N*C, H*W)), axis=1), (H, W))

        # # Switch row and column vector because of mpii notation
        # keypoint_preds = keypoint_preds[[1, 0], :]

        # # Transpose from 2xN*C to N*Cx2 and reshape into original batches to NxCx2
        # return keypoint_preds.transpose((1, 0)).reshape(N, C, 2)

        ################

        for i in range(len(heatmap_grid)):
            #inx = np.zeros((16, 2))
            a_list = []
            for a in range(len(heatmap_grid[i])):
                b_list = []
                for b in range(len(heatmap_grid[i][a])):
                    max = heatmap_grid[i][a][b].argmax().item()
            
                    x = (max // 64) * 4
                    y = (max % 64) * 4
            
                    b_list.append((x,y))
                a_list.append(b_list)
            max_idx = torch.from_numpy(np.asarray(a_list))
            break

        # print(max_idx.squeeze().shape)
        return max_idx

