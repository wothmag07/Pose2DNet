# Pose2DNet

**Pose2DNet** is a deep learning-based project designed to detect and analyze human poses in 2D images. The model is trained and evaluated using the widely-used **[MPII Human Pose Dataset](https://www.kaggle.com/datasets/harshpatel66/mpii-human-pose?select=mpii_human_pose.csv)**, which provides high-quality annotations for human pose estimation.

## Features

- **Keypoint Detection**: Detects multiple human body keypoints in 2D images.
- **Preprocessing Pipeline**: Handles image and annotation preprocessing to ensure robust training and evaluation.
- **Configurable Parameters**: Flexible inputs for custom datasets, annotations, and keypoints.

## Dataset

The **MPII Human Pose Dataset** is used to train and evaluate the model. This dataset contains around 25,000 images and over 40,000 annotated human poses, covering a wide range of activities.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Pose2DNet.git
   cd Pose2DNet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```bash
python pose2dnet.py -i /path/to/images -a /path/to/annotations.csv -k <number_of_keypoints>
```

### Arguments:
- `-i`, `--images`: Path to the directory containing images.
- `-a`, `--anno`: Path to the CSV file containing annotations.
- `-k`, `--keypts`: Number of keypoints annotated in the dataset.

### Example:
```bash
python pose2dnet.py -i ./data/images -a ./data/annotations.csv -k 16
```

## Results

Pose2DNet achieves competitive performance in human pose estimation, effectively detecting keypoints such as joints and limbs in diverse images with mAP score of 0.85
