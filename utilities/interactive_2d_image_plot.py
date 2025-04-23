import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

class Interactive2DImagePlot:
    def __init__(self, points, images, labels):
        if len(points) != len(images) or len(points) != len(labels):
            raise ValueError("points, images, and labels must be of the same length.")

        self.points = np.array(points)
        self.images = images
        self.labels = labels
        self.fig, self.ax = plt.subplots()

        # Convert labels to scalar values
        scalar_labels = [label.item() if hasattr(label, 'item') else label for label in labels]

        # Map labels to colors
        unique_labels = np.unique(scalar_labels)
        cmap = plt.get_cmap('viridis')
        self.colors = cmap(np.linspace(0, 1, len(unique_labels)))
        self.label_to_color = {label: self.colors[i] for i, label in enumerate(unique_labels)}

        # Create scatter plot with colors
        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], c=[self.label_to_color[label] for label in scalar_labels])
        self.annotation_box = None
        self.img_preview = None

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_hover(self, event):
        if event.inaxes != self.ax:
            return

        mouse_point = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.points - mouse_point, axis=1)
        closest_idx = np.argmin(dists)

        # Remove previous image
        if self.annotation_box:
            self.annotation_box.remove()
            self.annotation_box = None

        # Load and preview new image
        img = self.images[closest_idx]

        # Convert the tensor to a numpy array
        img_np = img.squeeze().numpy()  # Remove the channel dimension if it exists

        # Convert to PIL Image
        img_pil = Image.fromarray(img_np)

        # Resize the image
        img_resized = img_pil.resize((64, 64))

        # Convert back to numpy array for OffsetImage
        img_resized_np = np.asarray(img_resized)

        # Create the OffsetImage and AnnotationBbox
        imagebox = OffsetImage(img_resized_np, zoom=1.0)
        self.annotation_box = AnnotationBbox(imagebox, self.points[closest_idx], frameon=True)
        self.ax.add_artist(self.annotation_box)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

def interactive_2d_image_plot(points, images, labels):
    interactive_plot = Interactive2DImagePlot(points, images, labels)
    interactive_plot.show()
