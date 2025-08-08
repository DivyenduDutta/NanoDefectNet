import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from skimage import io

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger


def browse_and_display() -> None:
    """
    Open a file dialog to select an image and display it.

    """
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )

    if filepath:
        img = io.imread(filepath)

        LOGGER.info(f"img.shape: {img.shape}")
        LOGGER.info(f"img.dtype: {img.dtype}")

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Image: {filepath.split('/')[-1]}")
        plt.axis("off")
        plt.show()


root = tk.Tk()
root.title("Image Browser")
root.geometry("300x100")

browse_button = tk.Button(
    root, text="Browse Image", command=browse_and_display, height=2, width=20
)
browse_button.pack(pady=20)

root.mainloop()
