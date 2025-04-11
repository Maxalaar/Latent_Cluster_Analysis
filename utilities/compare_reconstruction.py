from pathlib import Path
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from architecture.autoencoder import Autoencoder

def compare_reconstruction(model: Autoencoder, samples: torch.Tensor, path: Path):
    """
    For each sample, create a composite image consisting of:
    - On the left: the reconstruction obtained from the autoencoder.
    - On the right: the original image.
    The generated images are saved in the directory path.
    """
    # Ensure that the main save directory exists.
    path.mkdir(parents=True, exist_ok=True)
    # Ensure that the subdirectory for reconstructions exists.
    reconstruction_directory = path / 'reconstruction'
    reconstruction_directory.mkdir(parents=True, exist_ok=True)

    # Preparation for conversion to a Pillow image.
    convert_to_pillow = ToPILImage()

    # Set the minimum dimensions for the composite image.
    MINIMUM_WIDTH = 256
    MINIMUM_HEIGHT = 256

    # Iterate over each sample. For every sample, the autoencoder is applied, and a composite image is created.
    for i, sample in enumerate(samples):
        # Prepare the batch as (1, channels, height, width) and move to the device of the autoencoder.
        sample_batch = sample.unsqueeze(0).to(model.device)
        with torch.no_grad():
            # Perform a forward pass through the autoencoder.
            encoded_sample = model.encode(sample_batch)
            reconstruction = model.decode(encoded_sample)

        # Move back to the CPU and remove the batch dimension.
        reconstruction_tensor = reconstruction.cpu().squeeze(0)
        original_tensor = sample_batch.cpu().squeeze(0)

        # Convert the tensors to Pillow images.
        # For one-channel images, mode "L" is sufficient.
        reconstruction_image = convert_to_pillow(reconstruction_tensor)
        original_image = convert_to_pillow(original_tensor)

        # Create the composite image.
        # The composite image has a width equal to the sum of both images widths and a height equal to the maximum height.
        total_width = reconstruction_image.width + original_image.width
        maximum_height = max(reconstruction_image.height, original_image.height)
        composite_image = Image.new('L', (total_width, maximum_height))
        # Paste the reconstruction on the left and the original on the right.
        composite_image.paste(reconstruction_image, (0, 0))
        composite_image.paste(original_image, (reconstruction_image.width, 0))

        # Check whether the composite image meets the minimum width or minimum height.
        # If not, compute the scale factor that makes both dimensions meet the threshold while preserving the ratio.
        if composite_image.width < MINIMUM_WIDTH or composite_image.height < MINIMUM_HEIGHT:
            scale_factor = max(MINIMUM_WIDTH / composite_image.width, MINIMUM_HEIGHT / composite_image.height)
            new_width = int(composite_image.width * scale_factor)
            new_height = int(composite_image.height * scale_factor)
            composite_image = composite_image.resize((new_width, new_height), Image.LANCZOS)

        # Save the composite image.
        composite_image.save(reconstruction_directory / f"sample_{i}.png")
