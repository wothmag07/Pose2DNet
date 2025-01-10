import os
import cv2
import pandas as pd
from multiprocessing import Pool
import time

input_dir = "D:\\Workspace\\2D-Pose-Estimation\\images\\"
output_dir = 'D:\\Workspace\\2D-Pose-Estimation\\resized-images\\'
num_processes = 4  # You can adjust this based on your system's capabilities

df = pd.read_csv(r'D:\Workspace\2D-Pose-Estimation\annotations\mpii_dataset.csv')


def resize_image(ix):
    img_name = df['NAME'].iloc[ix]
    image = cv2.imread(os.path.join(input_dir, img_name))
    old_height, old_width, _ = image.shape
    new_width, new_height = 224, 224
    resized_image = cv2.resize(image, (new_width, new_height))

    cv2.imwrite(os.path.join(output_dir, img_name), resized_image)
    
    return old_height, old_width

if __name__ == '__main__':
    start_time = time.time()  # Start time tracking
    pool = Pool(processes=num_processes)
    dimensions = pool.map(resize_image, range(len(df)))
    pool.close()
    pool.join()
    end_time = time.time()  # End time tracking

    print(f"Total time taken : {end_time - start_time}")
    
    # Update the resized DataFrame with image dimensions
    df['Height'], df['Width'] = zip(*dimensions)
    
    # Save the resized DataFrame
    df.to_csv(r'D:\Workspace\2D-Pose-Estimation\annotations\mpii_hw_dataset.csv', index=False)

print("All images resized and dimensions saved successfully!")
