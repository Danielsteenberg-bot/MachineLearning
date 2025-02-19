import matplotlib.pyplot as plt
import math

def visualize_dataset(dataset, class_names, num_images=9, figsize=(10, 10)):
    """
    Visualize a batch of images from a TensorFlow dataset.
    
    Parameters:
        dataset: A TensorFlow dataset that yields batches of images and labels.
        class_names: List of class names corresponding to the labels.
        num_images: Number of images to display (default is 9).
        figsize: Size of the figure (default is (10, 10)).
    """
    plt.figure(figsize=figsize)
    grid_size = math.ceil(math.sqrt(num_images))
    
    # Retrieve one batch from the dataset.
    for images, labels in dataset.take(1):
        for i in range(min(num_images, images.shape[0])):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()