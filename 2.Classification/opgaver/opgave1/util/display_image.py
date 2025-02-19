import matplotlib.pyplot as plt
import PIL
from pathlib import Path
from typing import List, Union

def display_images(image_paths: List[Union[str, Path]], 
                  title: str = 'Images', 
                  num_images: int = 2) -> None:
    """
    Display a set of images in a grid.
    
    Args:
        image_paths: List of paths to images
        title: Title for the figure
        num_images: Number of images to display 
    """
    if not image_paths:
        print("No images found to display!")
        return
    
    images_to_show = image_paths[:num_images]
    
    fig = plt.figure(figsize=(5*num_images, 5))
    fig.suptitle(title)
    
    for i, path in enumerate(images_to_show, 1):
        plt.subplot(1, num_images, i)
        img = PIL.Image.open(str(path))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Image {i}')
    
    plt.tight_layout()
    plt.show()