import itertools
import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd

input_dir = "D:\\Workspace\\Pose2DNet\\images\\"
output_dir = 'D:\\Workspace\\Pose2DNet\\resized-images\\'

df = pd.read_csv(r'D:\Workspace\Pose2DNet\annotations\mpii_hw_dataset.csv')

df_normalised = pd.DataFrame(columns=df.columns)

new_width, new_height = 224, 224

df_normalised[['NAME', 'Scale', 'Activity', 'Category']] = df[['NAME', 'Scale', 'Activity', 'Category']]

df_normalised['Height'] = 224
df_normalised['Width'] = 224

df_normalised['scale_X'] = df_normalised['Width'] / df['Width']

df_normalised['scale_Y'] = df_normalised['Height'] / df['Height']

# Get the column names for X and Y coordinates
x_columns = [col for col in df.columns if col.endswith('_X')]
y_columns = [col for col in df.columns if col.endswith('_Y')]


# Iterate over X and Y columns and calculate scaled values
for x_col, y_col in zip(x_columns, y_columns):
    
    df_normalised[x_col] = (df[x_col] * df_normalised['scale_X'].where(df[x_col] != -1, 1)).astype(int)
    df_normalised[y_col] = (df[y_col] * df_normalised['scale_Y'].where(df[y_col] != -1, 1)).astype(int)


# print(df_normalised.head(2))

# print(df.head(2))
    
keypoints_df = df_normalised.iloc[:, 1:33].copy()
    
keypoints_df.to_csv('annotations/mpii_keypoints.csv', index=False)

'''
def image(ix):
    
    img_name = df_normalised['NAME'][ix]
    print(img_name)
    
    image = cv2.imread(os.path.join(output_dir, img_name))
    print(image.shape)
    
    # Display the image
    plt.imshow(image)

    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y']) 

    for i in range(1, 33, 2):
        col_x = df_normalised.columns[i]
        x = df_normalised.iloc[ix, i] 
        col_y = df_normalised.columns[i + 1]
        y = df_normalised.iloc[ix, i + 1]
        print(col_x, x, col_y, y)
        color = next(colors)
        plt.plot(x, y, 'o', c=color, markersize=3)  # Plot keypoints as circles
    
    plt.show()

image(100)

'''