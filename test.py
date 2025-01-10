import torch
from torchvision import transforms
from PIL import Image
from model1 import StackedHourGlass
from process import ToTensor, Resize, HeatmapToKeypointsConverter
import matplotlib.pyplot as plt
import cv2


# Load the saved model
model = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=16)
model.load_state_dict(torch.load("model_epoch_10.pt", map_location=torch.device('cpu')))  # Load model weights
model.eval()

# Load and preprocess the image
image_path = "images/000033016.jpg"  # Path to your test image
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to the input size of your model
    transforms.ToTensor()          # Convert to tensor
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_batch)
    
heatmap2keypoints = HeatmapToKeypointsConverter()

keypoints = heatmap2keypoints(output)

print(keypoints)


def plot_keypoints(image, keypoints, save_path="plot_with_keypoints.png"):
    """
    Plot image with keypoints overlaid.

    Args:
    - image (numpy.ndarray): Input image.
    - keypoints (list): Detected keypoints. List of (x, y) coordinate pairs.
    - save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    plt.imshow(image)
    for keypoint in keypoints:
        x, y = keypoint
        plt.scatter(x, y, color='red', s=10)  # Plot each keypoint
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Resize the image using OpenCV and then convert it to a PIL Image
image_data = cv2.imread(image_path)
image_resized = cv2.resize(image_data, (256, 256))
image_resized_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

# Plot keypoints on the image
plot_keypoints(image_resized_pil, keypoints[0])
