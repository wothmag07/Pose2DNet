import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
import os

def keypoints_to_heatmap(keypoints, heatmap_shape=(64, 64), sigma=1.0):
    heatmap = np.zeros(heatmap_shape)

    for x, y in keypoints:
        # Scale down the keypoints to fit into the heatmap dimensions
        x_scaled = int(x * heatmap_shape[1] / 255)
        y_scaled = int(y * heatmap_shape[0] / 255)

        # Generate a Gaussian kernel centered at the scaled keypoint
        kernel_size = int(sigma * 3) * 2 + 1  # Ensure kernel size is odd
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = math.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))

        # Calculate bounding box for kernel placement
        top = max(0, y_scaled - center)
        bottom = min(heatmap_shape[0], y_scaled - center + kernel_size)
        left = max(0, x_scaled - center)
        right = min(heatmap_shape[1], x_scaled - center + kernel_size)

        # Update heatmap within bounding box
        heatmap[top:bottom, left:right] += kernel[:bottom-top, :right-left]

    # Normalize the heatmap values
    heatmap /= np.max(heatmap)

    return heatmap

def plot_image_with_heatmap(image, keypoints, image_dir):

    # Overlay keypoints on the resized image
    keypoints = keypoints.squeeze(0)
    # print(keypoints)
    # Overlay keypoints on the image
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green circle for keypoints
    
    heatmaps = keypoints_to_heatmap(keypoints)

    # Display the resized image and the heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title('Resized Image with keypoints')

    plt.subplot(1, 2, 2)
    plt.imshow(heatmaps, cmap='hot', interpolation='nearest')
    plt.title('Heatmap')
    plt.savefig(os.path.join("outputs/", 'image-heatmap.png'), bbox_inches='tight')
    plt.show()




