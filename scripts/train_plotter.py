import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
from glob import glob

def plot_yolo_bbox_dataset(dataset_path, num_samples=4, class_names=None, ext="*.tif"):
    """
    Plot random samples from a YOLO dataset
    
    Args:
        dataset_path: Path to dataset folder (should contain 'images' and 'labels' subfolders)
        num_samples: Number of random samples to plot
        class_names: List of class names (optional)
    """
    image_files = glob(dataset_path + f'/images/{ext}')
    
    # Select random samples
    num_samples = min(num_samples, len(image_files))
    selected_samples = random.sample(image_files, num_samples)
    
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes_all = plt.subplots(rows, cols, figsize=(15, 5*rows))

    # Ensure axes_all is always an iterable of Axes objects (flatten if it's an array)
    if isinstance(axes_all, np.ndarray):
        axes_all = axes_all.flatten()
    elif not isinstance(axes_all, (list, tuple)): # Handles case where plt.subplots returns a single Axes object
        axes_all = [axes_all]

    # Select only the number of axes corresponding to num_samples
    axes = axes_all[:len(selected_samples)]
    
    for idx, image_path in enumerate(selected_samples):
        if idx >= num_samples:
            break
            
        # Get corresponding label file
        label_path = image_path.replace("/images/", "/labels/").replace(".tif", ".txt")
        
        # Read and plot image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape
        
        axes[idx].imshow(image)
        
        # Plot bounding boxes if label exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                data = line.strip().split()
                if len(data) >= 5:
                    class_id = int(data[0])
                    x_center = float(data[1]) * img_width
                    y_center = float(data[2]) * img_height
                    width = float(data[3]) * img_width
                    height = float(data[4]) * img_height
                    
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    
                    rect = plt.Rectangle((x1, y1), width, height, 
                                       fill=False, edgecolor='red', linewidth=2)
                    axes[idx].add_patch(rect)
                    
                    if class_names and class_id < len(class_names):
                        label = class_names[class_id]
                        axes[idx].text(x1, y1-5, label, color='white', fontsize=8,
                                     bbox=dict(facecolor='red', alpha=0.7))
        
        axes[idx].set_title(f'Sample {idx+1}: {os.path.split(image_path)[1]}')
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


  
  
def plot_yolo_segmentation_dataset(dataset_path, num_samples=9, class_names=None, ext="*.tif"):
    """
    Plot random samples from YOLO segmentation dataset with their masks

    Args:
        dataset_path: Path to dataset (should contain 'images' and 'labels' folders)
        num_samples: Number of samples to plot
        class_names: List of class names (optional)
    """
    image_files = list(glob(dataset_path + '/images/'+ ext))
    labels_files = [file.replace('/images/', '/labels/').replace(".tif", ".txt") for file in image_files]

    print(f"{len(image_files)} images and {len(labels_files)} labels found in the file directory")

    # Get all image files
    # image_files = list(images_path.glob(ext))

    if len(image_files) == 0:
        print(f"No images found in {dataset_path}")
        return

    # Randomly select samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    # Create subplot grid
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes_all = plt.subplots(rows, cols, figsize=(15, 5*rows))

    # Ensure axes_all is always an iterable of Axes objects (flatten if it's an array)
    if isinstance(axes_all, np.ndarray):
        axes_all = axes_all.flatten()
    elif not isinstance(axes_all, (list, tuple)): # Handles case where plt.subplots returns a single Axes object
        axes_all = [axes_all]

    # Select only the number of axes corresponding to num_samples
    axes = axes_all[:len(samples)]

    for idx, img_path in enumerate(samples):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # Load corresponding label file
        label_path = img_path.replace('/images/', '/labels/').replace(".tif", ".txt")

        # Create mask overlay
        mask_overlay = image.copy()

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                # Parse YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                class_id = int(parts[0])

                # Get polygon points
                points = np.array(parts[1:], dtype=np.float32)
                points = points.reshape(-1, 2)

                # Convert normalized coordinates to pixel coordinates
                points[:, 0] *= img_w
                points[:, 1] *= img_h
                points = points.astype(np.int32)

                # Create mask for this polygon
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 1)

                # Generate random color for overlay
                color = [random.randint(0, 255) for _ in range(3)]

                # Apply colored overlay with transparency
                colored_mask = np.zeros_like(image)
                colored_mask[mask == 1] = color
                mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, 0.5, 0)

                # Draw polygon outline
                cv2.polylines(mask_overlay, [points], True, color, 2)

                # Add class label if names are provided
                if class_names and class_id < len(class_names):
                    # Find top-left point for text
                    x, y = points.min(axis=0)
                    cv2.putText(mask_overlay, class_names[class_id],
                               (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display image
        axes[idx].imshow(mask_overlay)
        axes[idx].set_title(f'Sample: {Path(img_path).name}', fontsize=10)
        axes[idx].axis('off')

    # Hide empty subplots that were created by plt.subplots but not used
    for idx in range(len(samples), len(axes_all)):
        axes_all[idx].axis('off')

    plt.tight_layout()
    plt.show()



def plot_yolo_obb_dataset(dataset_path, class_names=None, num_samples=5, ext="*.tif"):
    """
    Plot YOLO Oriented Bounding Box dataset samples

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing OBB label files (.txt)
        class_names: List of class names (optional)
        num_samples: Number of samples to display
        save_fig: Path to save the figure (optional)
    """

    # Get list of image files

    image_files = list(glob(dataset_path + '/images/'+ ext))

    # Randomly select samples
    if len(image_files) > num_samples:
        import random
        image_files = random.sample(image_files, num_samples)

    # Create subplot grid
    n_cols = min(3, num_samples)
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    # Process each image
    for idx, img_path in enumerate(image_files[:num_samples]):
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Get corresponding label file
        label_path = img_path.replace('/images/', '/labels/').replace(".tif", ".txt")

        # Plot image
        axes[idx].imshow(img)
        axes[idx].set_title(f"Sample {idx+1}: {os.path.split(img_path)[1]}")

        # Draw bounding boxes if label exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(label_path)

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9:  # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                    class_id = int(parts[0])

                    # Get normalized coordinates and convert to absolute
                    points = np.array([float(p) for p in parts[1:9]]).reshape(4, 2)
                    points[:, 0] *= w
                    points[:, 1] *= h
                    points = points.astype(np.int32)

                    # Get color for class
                    color = plt.cm.tab10(class_id % 10)[:3]

                    # Draw polygon
                    polygon = np.array([points], np.int32)
                    cv2.polylines(img, polygon, isClosed=True,
                                 color=[int(c*255) for c in color], thickness=2)

                    # Add class label
                    if class_names and class_id < len(class_names):
                        label = class_names[class_id]
                    else:
                        label = f"Class {class_id}"

                    # Calculate centroid for label position
                    centroid = points.mean(axis=0).astype(np.int32)
                    cv2.putText(img, label, (centroid[0]-10, centroid[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              [int(c*255) for c in color], 1, cv2.LINE_AA)

            # Update image with drawn boxes
            axes[idx].imshow(img)
            axes[idx].set_title(f'Sample: {os.path.split(img_path)[1]}', fontsize=10)
            axes[idx].axis('off')

    # Hide empty subplots
    for idx in range(len(image_files[:num_samples]), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    plt.show()    

