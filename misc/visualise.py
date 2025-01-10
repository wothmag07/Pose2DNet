import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Input and output directories
input_dir = "/nfs/stak/users/arulmozg/hpc-share/2D-Pose-Estimation/images"
output_dir = '/nfs/stak/users/arulmozg/hpc-share/2D-Pose-Estimation/outputs'


def image(ix, output_dir, df):
    img_name = df['NAME'][ix]
    print(img_name)
    
    image = cv2.imread(os.path.join(input_dir, img_name))
    print(image.shape)
    
    # Display the image
    plt.imshow(image)

    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y']) 

    for i in range(1, 33, 2):
        col_x = df.columns[i]
        x = df.iloc[ix, i] 
        col_y = df.columns[i + 1]
        y = df.iloc[ix, i + 1]
        print(col_x, x, col_y, y)
        color = next(colors)
        plt.plot(x, y, 'o', c=color, markersize=3)  # Plot keypoints as circles
    
    # Save the image
    plt.savefig(os.path.join(output_dir, f"{img_name}_keypoints.jpg"))
    
    plt.show()

# Example usage:
df = pd.read_csv('annotations/mpii_dataset.csv')
image(0, output_dir, df)
