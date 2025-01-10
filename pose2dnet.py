import argparse
from torchvision import transforms
import os
import numpy as np

from process import MPIIDataset, ToTensor, Resize, HeatmapGenerator
# from model import PoseEstimationModel
from train import train_model
from model import StackedHourGlass
import matplotlib.pyplot as plt
from visualise1 import plot_image_with_heatmap

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--images", required=True, type=str,
                                        help="Specify path to images directory.")
    parser.add_argument('-a', "--anno", required=True, type=str,
                                        help="Specify path to annotations file (csv).")
    parser.add_argument('-k', "--keypts", required=True, type=int,
                                        help="Specify number of keypoints that is annotated.")
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    num_keypoints = args.keypts

    image_size = 256

    data_transform = transforms.Compose([
                                            Resize(image_size)  # Resize to 256x256
                                        
                                        ])
    dataset = MPIIDataset(root_dir=args.images,
                        annotation_file=args.anno,
                        transform=data_transform)

    img_dir = 'outputs/'
    
    kp = dataset[0]['keypoints'].reshape(1,16,2)
    img = dataset[0]['image']

    plot_image_with_heatmap(img, kp, img_dir)


    heatmap = HeatmapGenerator(64,16)

    kh = heatmap(kp)

    print(kh.shape)

    #Visualize the grid of heatmaps and save the plot
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(16):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(kh[0, i], cmap='hot', interpolation='nearest')
        axs[row, col].set_title(f'Joint {i+1}')
        axs[row, col].axis('off')
    plt.savefig(os.path.join("outputs/", 'heatmap_grid.png'), bbox_inches='tight')
    plt.close()

    print("Training Phase")

    # model = PoseEstimationModel(image_size=image_size, num_keypoints=num_keypoints)
    model = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=num_keypoints)
    train_model(model, dataset, batch_size=64, num_epochs=10, learning_rate=1e-3, device='cuda')

'''
# Create a new figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize an empty heatmap to aggregate all keypoints
all_keypoints_heatmap = np.zeros_like(kh[0, 0])

# Aggregate all keypoints into a single heatmap
for i in range(16):
    all_keypoints_heatmap += kh[0, i]

# Plot the aggregated heatmap
ax.imshow(all_keypoints_heatmap, cmap='hot', interpolation='nearest', alpha=0.5)
ax.set_title("All Keypoints")
ax.axis('off')

plt.savefig(os.path.join("outputs/", 'all_keypoints.png'), bbox_inches='tight')
plt.close()
'''


    
    

