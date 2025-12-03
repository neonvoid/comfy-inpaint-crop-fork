import comfy.utils
import comfy.model_management
import folder_paths
import math
import nodes
import torch
import time
import random
import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, label as scipy_label
from scipy.optimize import linear_sum_assignment
import json


def gaussian_smooth_1d(values, window_size):
    """Apply Gaussian smoothing to 1D sequence for crop stabilization"""
    if len(values) <= 1:
        return values
    sigma = window_size / 4.0  # ~95% of weight within window
    smoothed = gaussian_filter1d(np.array(values, dtype=float), sigma, mode='nearest')
    return smoothed.tolist()


def median_filter_1d(values, window_size):
    """Apply median filter to 1D sequence for crop stabilization"""
    if len(values) <= 1:
        return values
    filtered = median_filter(np.array(values), size=window_size, mode='nearest')
    return filtered.tolist()


def print_gpu_info():
    """Print GPU availability and device information"""
    print("\n[DEBUG] ========== GPU/Device Information ==========")
    print(f"[DEBUG] PyTorch version: {torch.__version__}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA version: {torch.version.cuda}")
        print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[DEBUG] Device name: {torch.cuda.get_device_name(0)}")
        print(f"[DEBUG] Device count: {torch.cuda.device_count()}")
        # Memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[DEBUG] GPU memory allocated: {allocated:.2f} GB")
        print(f"[DEBUG] GPU memory reserved: {reserved:.2f} GB")
    else:
        print("[DEBUG] WARNING: CUDA not available! All operations will run on CPU.")
    print("[DEBUG] ================================================\n")


def rescale_i(samples, width, height, algorithm: str):
    start_time = time.time()
    device = samples.device
    print(f"[DEBUG] rescale_i: Resizing image to {width}x{height} using {algorithm}, device={device}")

    # Map PIL algorithm names to PyTorch modes
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',  # Lanczos not directly supported, use bicubic
        'box': 'area',  # Box filter maps to area mode
        'hamming': 'bilinear',  # Hamming not directly supported, use bilinear
    }

    mode = algorithm_map.get(algorithm.lower(), 'bilinear')

    # Convert from [B, H, W, C] to [B, C, H, W] for interpolate
    samples = samples.movedim(-1, 1)

    gpu_start = time.time()
    # Use PyTorch interpolate - stays on GPU
    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] rescale_i: GPU-only resize took {gpu_time:.4f}s (previously CPU)")

    # Convert back to [B, H, W, C]
    samples = samples.movedim(1, -1)

    total_time = time.time() - start_time
    print(f"[DEBUG] rescale_i: Total time {total_time:.4f}s")
    return samples


def rescale_m(samples, width, height, algorithm: str):
    start_time = time.time()
    device = samples.device
    print(f"[DEBUG] rescale_m: Resizing mask to {width}x{height} using {algorithm}, device={device}")

    # Map PIL algorithm names to PyTorch modes
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',  # Lanczos not directly supported, use bicubic
        'box': 'area',  # Box filter maps to area mode
        'hamming': 'bilinear',  # Hamming not directly supported, use bilinear
    }

    mode = algorithm_map.get(algorithm.lower(), 'bilinear')

    # Add channel dimension for interpolate [B, H, W] -> [B, 1, H, W]
    samples = samples.unsqueeze(1)

    gpu_start = time.time()
    # Use PyTorch interpolate - stays on GPU
    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] rescale_m: GPU-only resize took {gpu_time:.4f}s (previously CPU)")

    # Remove channel dimension [B, 1, H, W] -> [B, H, W]
    samples = samples.squeeze(1)

    total_time = time.time() - start_time
    print(f"[DEBUG] rescale_m: Total time {total_time:.4f}s")
    return samples


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height

        scale_factor = max(scale_factor_min_width, scale_factor_min_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        
        assert preresize_min_width <= target_width <= preresize_max_width, \
            f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
        assert preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)

        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

    return image, mask, optional_context_mask


def binary_dilation_torch(mask, kernel_size=3):
    """Perform binary dilation using max pooling."""
    # Add channel dimension [B, H, W] -> [B, 1, H, W]
    mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
    padding = kernel_size // 2
    dilated = torch.nn.functional.max_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )
    return dilated.squeeze(1) if dilated.size(1) == 1 else dilated


def binary_erosion_torch(mask, kernel_size=3):
    """Perform binary erosion using negative dilation."""
    # Erosion = 1 - dilation(1 - mask)
    inverted = 1.0 - mask
    dilated = binary_dilation_torch(inverted, kernel_size)
    eroded = 1.0 - dilated
    return eroded


def binary_closing_torch(mask, kernel_size=3):
    """Perform binary closing (dilation followed by erosion)."""
    dilated = binary_dilation_torch(mask, kernel_size)
    closed = binary_erosion_torch(dilated, kernel_size)
    return closed


def binary_fill_holes_torch(mask):
    """Fill holes in binary mask using morphological reconstruction.
    This is a simplified version that works well for most cases."""
    # Create a seed that is 0 everywhere except the borders (which are inverted from the mask)
    b, h, w = mask.shape
    seed = torch.zeros_like(mask)

    # Set borders to inverted mask values
    seed[:, 0, :] = 1.0 - mask[:, 0, :]
    seed[:, -1, :] = 1.0 - mask[:, -1, :]
    seed[:, :, 0] = 1.0 - mask[:, :, 0]
    seed[:, :, -1] = 1.0 - mask[:, :, -1]

    # Iterative dilation with mask constraint (reconstruction by dilation)
    for _ in range(max(h, w)):  # Enough iterations to propagate through entire image
        dilated_seed = binary_dilation_torch(seed, kernel_size=3)
        # Constrain to inverted mask
        new_seed = torch.minimum(dilated_seed, 1.0 - mask)
        if torch.equal(new_seed, seed):
            break
        seed = new_seed

    # The filled mask is the original mask OR the inverse of the reconstruction
    filled = torch.maximum(mask, 1.0 - seed)
    return filled


def fillholes_iterative_hipass_fill_m(samples):
    start_time = time.time()
    device = samples.device
    print(f"[DEBUG] fillholes_iterative_hipass_fill_m: Starting hole filling on device={device}")

    thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # Keep everything on GPU
    mask = samples.squeeze(0)
    # Add batch dimension if needed [H, W] -> [1, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    gpu_start = time.time()
    for threshold in thresholds:
        # Threshold the mask
        thresholded_mask = (mask >= threshold).float()

        # Apply binary closing
        closed_mask = binary_closing_torch(thresholded_mask, kernel_size=3)

        # Fill holes
        filled_mask = binary_fill_holes_torch(closed_mask)

        # Update mask with filled values
        mask = torch.maximum(mask, filled_mask * threshold)

    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] fillholes_iterative_hipass_fill_m: GPU-only operations took {gpu_time:.4f}s (previously CPU)")

    total_time = time.time() - start_time
    print(f"[DEBUG] fillholes_iterative_hipass_fill_m: Total time {total_time:.4f}s")
    return mask.unsqueeze(0) if mask.dim() == 2 else mask


def hipassfilter_m(samples, threshold):
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def expand_m(mask, pixels):
    start_time = time.time()
    device = mask.device
    print(f"[DEBUG] expand_m: Expanding mask by {pixels} pixels, device={device}")

    sigma = pixels / 4
    kernel_size = math.ceil(sigma * 1.5 + 1)

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    gpu_start = time.time()

    # Add channel dimension for max_pool2d [B, H, W] -> [B, 1, H, W]
    mask = mask.unsqueeze(1)

    # Grey dilation is equivalent to max pooling
    # Apply max pooling with appropriate padding to expand the mask
    padding = kernel_size // 2
    dilated_mask = torch.nn.functional.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    # Remove channel dimension [B, 1, H, W] -> [B, H, W]
    dilated_mask = dilated_mask.squeeze(1)

    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] expand_m: GPU-only max_pool2d (dilation) took {gpu_time:.4f}s (previously CPU)")

    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)

    total_time = time.time() - start_time
    print(f"[DEBUG] expand_m: Total time {total_time:.4f}s")
    return dilated_mask


def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask


def blur_m(samples, pixels):
    start_time = time.time()
    device = samples.device
    print(f"[DEBUG] blur_m: Blurring mask with {pixels} pixels, device={device}")

    sigma = pixels / 4

    # Create Gaussian kernel on GPU
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    gpu_start = time.time()

    # Generate 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Add dimensions for conv2d: [1, 1, kernel_size, 1] and [1, 1, 1, kernel_size]
    kernel_h = gauss.view(1, 1, kernel_size, 1)
    kernel_w = gauss.view(1, 1, 1, kernel_size)

    # Add channel dimension for convolution [B, H, W] -> [B, 1, H, W]
    mask = samples.unsqueeze(1)

    # Apply separable Gaussian blur (horizontal then vertical)
    padding = kernel_size // 2
    mask = torch.nn.functional.conv2d(mask, kernel_h, padding=(padding, 0))
    mask = torch.nn.functional.conv2d(mask, kernel_w, padding=(0, padding))

    # Remove channel dimension [B, 1, H, W] -> [B, H, W]
    mask = mask.squeeze(1)

    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] blur_m: GPU-only gaussian blur took {gpu_time:.4f}s (previously CPU)")

    mask = torch.clamp(mask, 0.0, 1.0)

    total_time = time.time() - start_time
    print(f"[DEBUG] blur_m: Total time {total_time:.4f}s")
    return mask


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    start_time = time.time()
    device = image.device
    print(f"[DEBUG] extend_imm: Extending image for outpainting, device={device}")

    B, H, W, C = image.shape

    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

    gpu_start = time.time()
    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)
    gpu_time = time.time() - gpu_start
    print(f"[DEBUG] extend_imm: GPU tensor allocation took {gpu_time:.4f}s")

    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding

    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)

    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)

    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    total_time = time.time() - start_time
    print(f"[DEBUG] extend_imm: Total time {total_time:.4f}s")
    return expanded_image, expanded_mask, expanded_optional_context_mask


def debug_context_location_in_image(image, x, y, w, h):
    debug_image = image.clone()
    debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
    return debug_image


def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]

    # Compute intended growth in each direction
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))

    # Try to grow left, but clamp at 0
    new_x = x - grow_left
    if new_x < 0:
        new_x = 0

    # Try to grow up, but clamp at 0
    new_y = y - grow_up
    if new_y < 0:
        new_y = 0

    # Right edge
    new_x2 = x + w + grow_right
    if new_x2 > img_w:
        new_x2 = img_w

    # Bottom edge
    new_y2 = y + h + grow_down
    if new_y2 > img_h:
        new_y2 = img_h

    # New width and height
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    # Extract the context
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]

    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    start_time = time.time()
    device = image.device
    print(f"[DEBUG] crop_magic_im: Starting crop operation, device={device}")
    print(f"[DEBUG] crop_magic_im: Context area ({x}, {y}, {w}, {h}), target size {target_w}x{target_h}")

    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.

    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h

    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    if context_aspect_ratio < target_aspect_ratio:
        # Grow width to meet aspect ratio
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y

        # Adjust new_x to keep within bounds
        if new_x < 0:
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow

    else:
        # Grow height to meet aspect ratio
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

        # Adjust new_y to keep within bounds
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow

    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

    expanded_image_w = image_w
    expanded_image_h = image_h

    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask

    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h

    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    print(f"[DEBUG] crop_magic_im: Resizing from {ctc_w}x{ctc_h} to {target_w}x{target_h}")
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

    total_time = time.time() - start_time
    print(f"[DEBUG] crop_magic_im: Total time {total_time:.4f}s")
    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    start_time = time.time()
    device = canvas_image.device
    print(f"[DEBUG] stitch_magic_im: Starting stitch operation, device={device}")

    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:  # Upscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:  # Downscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [1, H, W, 1]

    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

    total_time = time.time() - start_time
    print(f"[DEBUG] stitch_magic_im: Total time {total_time:.4f}s")
    return output_image


def analyze_and_track_masks(mask_batch, min_size=500, max_dist=150, threshold=0.5):
    """
    Analyze mask batch: detect connected regions, track across frames, assign persistent IDs.

    Args:
        mask_batch: Tensor [B, H, W] - batch of masks
        min_size: Minimum pixel count to consider a region (filter noise)
        max_dist: Maximum centroid distance for tracking between frames
        threshold: Binarization threshold for mask

    Returns:
        tracked_masks: Tensor [B, H, W] with persistent player IDs as values
        tracking_info: Dict with tracking statistics
    """
    start_time = time.time()
    device = mask_batch.device
    B, H, W = mask_batch.shape

    print(f"[DEBUG] analyze_and_track_masks: Processing {B} frames, min_size={min_size}, max_dist={max_dist}")

    tracked_masks = torch.zeros_like(mask_batch)

    next_player_id = 1
    prev_regions = []  # List of {player_id, centroid, area}
    player_timelines = {}  # player_id -> list of frame indices
    frame_player_counts = []  # Number of players per frame

    for frame_idx in range(B):
        # Binary threshold and convert to numpy for scipy
        binary = (mask_batch[frame_idx] > threshold).cpu().numpy()

        # Connected component labeling
        labeled, num_features = scipy_label(binary)

        # Extract current frame regions
        curr_regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            area = region_mask.sum()
            if area >= min_size:
                ys, xs = np.where(region_mask)
                centroid = (ys.mean(), xs.mean())
                curr_regions.append({
                    'local_label': i,
                    'centroid': centroid,
                    'area': int(area),
                    'mask': region_mask
                })

        # Match to previous frame using Hungarian algorithm on centroid distance
        if prev_regions and curr_regions:
            cost_matrix = np.zeros((len(prev_regions), len(curr_regions)))
            for i, prev in enumerate(prev_regions):
                for j, curr in enumerate(curr_regions):
                    dist = np.sqrt((prev['centroid'][0] - curr['centroid'][0])**2 +
                                  (prev['centroid'][1] - curr['centroid'][1])**2)
                    cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_curr = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_dist:
                    # Assign previous ID to matched region
                    curr_regions[c]['player_id'] = prev_regions[r]['player_id']
                    matched_curr.add(c)

            # Unmatched current regions get new IDs
            for c, region in enumerate(curr_regions):
                if c not in matched_curr:
                    region['player_id'] = next_player_id
                    next_player_id += 1
        else:
            # First frame or no previous regions - assign new IDs
            for region in curr_regions:
                region['player_id'] = next_player_id
                next_player_id += 1

        # Build output mask for this frame
        frame_output = np.zeros((H, W), dtype=np.float32)
        for region in curr_regions:
            frame_output[region['mask']] = region['player_id']

            # Track timeline
            pid = region['player_id']
            if pid not in player_timelines:
                player_timelines[pid] = []
            player_timelines[pid].append(frame_idx)

        tracked_masks[frame_idx] = torch.from_numpy(frame_output).to(device)
        frame_player_counts.append(len(curr_regions))

        # Update prev_regions for next frame (without the mask to save memory)
        prev_regions = [{'player_id': r['player_id'], 'centroid': r['centroid'], 'area': r['area']}
                       for r in curr_regions]

    # Build tracking info
    total_players = next_player_id - 1
    max_concurrent = max(frame_player_counts) if frame_player_counts else 0

    tracking_info = {
        'total_players': total_players,
        'max_concurrent': max_concurrent,
        'frame_count': B,
        'player_timelines': {str(k): v for k, v in player_timelines.items()},  # JSON-safe keys
        'frame_player_counts': frame_player_counts
    }

    elapsed = time.time() - start_time
    print(f"[DEBUG] analyze_and_track_masks: Found {total_players} unique players, max {max_concurrent} concurrent, took {elapsed:.4f}s")

    return tracked_masks, tracking_info


def limit_regions_per_frame(mask_batch, max_regions=4, batch_index=0, min_size=500, mode="largest"):
    """
    For each frame, detect connected regions and keep only the selected batch.

    Args:
        mask_batch: Tensor [B, H, W] - batch of masks
        max_regions: Maximum regions to keep per frame
        batch_index: Which batch of regions (0=first N, 1=next N, etc.)
        min_size: Minimum pixel count to consider a region
        mode: Selection mode - "largest", "smallest", "leftmost", "rightmost"

    Returns:
        output_masks: Tensor [B, H, W] with only selected regions (binary)
        max_batches: Maximum batches needed to cover all regions
        regions_per_frame: List of region counts per frame
    """
    start_time = time.time()
    device = mask_batch.device
    B, H, W = mask_batch.shape

    print(f"[DEBUG] limit_regions_per_frame: Processing {B} frames, max_regions={max_regions}, batch_index={batch_index}, mode={mode}")

    output_masks = torch.zeros_like(mask_batch)
    max_regions_in_any_frame = 0
    regions_per_frame = []

    for frame_idx in range(B):
        binary = (mask_batch[frame_idx] > 0.5).cpu().numpy()
        labeled, num_features = scipy_label(binary)

        # Extract regions with properties
        regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            area = region_mask.sum()
            if area >= min_size:
                ys, xs = np.where(region_mask)
                regions.append({
                    'mask': region_mask,
                    'area': int(area),
                    'center_x': float(xs.mean()),
                    'center_y': float(ys.mean())
                })

        regions_per_frame.append(len(regions))
        max_regions_in_any_frame = max(max_regions_in_any_frame, len(regions))

        # Sort by selection mode
        if mode == "largest":
            regions.sort(key=lambda r: r['area'], reverse=True)
        elif mode == "smallest":
            regions.sort(key=lambda r: r['area'])
        elif mode == "leftmost":
            regions.sort(key=lambda r: r['center_x'])
        elif mode == "rightmost":
            regions.sort(key=lambda r: r['center_x'], reverse=True)

        # Select batch
        start_idx = batch_index * max_regions
        end_idx = start_idx + max_regions
        selected = regions[start_idx:end_idx]

        # Build output mask
        frame_output = np.zeros((H, W), dtype=np.float32)
        for region in selected:
            frame_output[region['mask']] = 1.0

        output_masks[frame_idx] = torch.from_numpy(frame_output).to(device)

    max_batches = (max_regions_in_any_frame + max_regions - 1) // max_regions if max_regions_in_any_frame > 0 else 1

    elapsed = time.time() - start_time
    print(f"[DEBUG] limit_regions_per_frame: Max {max_regions_in_any_frame} regions in any frame, {max_batches} batches needed, took {elapsed:.4f}s")

    return output_masks, max_batches, regions_per_frame


class InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Required inputs
                "image": ("IMAGE",),

                # Resize algorithms
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),

                # Pre-resize input image
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Resize the original image before processing."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),

                # Mask manipulation
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Mark as masked any areas fully enclosed by mask."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by a certain amount of pixels before processing."}),
                "mask_invert": ("BOOLEAN", {"default": False,"tooltip": "Invert mask so that anything masked will be kept."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "How many pixels to blend into the original image."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Ignore mask values lower than this value."}),

                # Extend image for outpainting
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image for outpainting."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),

                # Context
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Grow the context area from the mask by a certain factor in every direction. For example, 1.5 grabs extra 50% up, down, left, and right as context."}),

                # Output
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Force a specific resolution for sampling."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
           },
           "optional": {
                # Optional inputs
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),

                # Crop stabilization for video/animation (off by default)
                "stabilize_crop": ("BOOLEAN", {"default": False, "tooltip": "Enable temporal stabilization of crop coordinates for smoother video/animation results"}),
                "stabilization_mode": (["smooth", "lock_first", "lock_largest", "median"], {"default": "smooth", "tooltip": "smooth=gaussian blur, lock_first=use first frame size, lock_largest=use max size, median=median filter"}),
                "smooth_window": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2, "tooltip": "Window size for smoothing (higher=smoother). Only odd numbers."}),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    DESCRIPTION = "Crops an image around a mask for inpainting, the optional context mask defines an extra area to keep for the context."


    # Remove the following # to turn on debug mode (extra outputs, print statements)
    #'''
    DEBUG_MODE = False
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")

    '''
    
    DEBUG_MODE = True # TODO
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK",
        # DEBUG
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask",
        # DEBUG
        "DEBUG_preresize_image",
        "DEBUG_preresize_mask",
        "DEBUG_fillholes_mask",
        "DEBUG_expand_mask",
        "DEBUG_invert_mask",
        "DEBUG_blur_mask",
        "DEBUG_hipassfilter_mask",
        "DEBUG_extend_image",
        "DEBUG_extend_mask",
        "DEBUG_context_from_mask",
        "DEBUG_context_from_mask_location",
        "DEBUG_context_expand",
        "DEBUG_context_expand_location",
        "DEBUG_context_with_context_mask",
        "DEBUG_context_with_context_mask_location",
        "DEBUG_context_to_target",
        "DEBUG_context_to_target_location",
        "DEBUG_context_to_target_image",
        "DEBUG_context_to_target_mask",
        "DEBUG_canvas_image",
        "DEBUG_orig_in_canvas_location",
        "DEBUG_cropped_in_canvas_location",
        "DEBUG_cropped_mask_blend",
    )
    #'''

    def analyze_batch_coordinates(self, mask_batch):
        """First pass: extract bounding boxes for all frames for stabilization"""
        coords = []
        for b in range(mask_batch.shape[0]):
            one_mask = mask_batch[b].unsqueeze(0)
            mask_sum = torch.count_nonzero(one_mask)
            if mask_sum == 0:
                coords.append(None)  # Will be skipped
            else:
                _, x, y, w, h = findcontextarea_m(one_mask)
                coords.append({'x': x, 'y': y, 'w': w, 'h': h})
        return coords

    def stabilize_coordinates(self, coords, mode, window_size):
        """Apply stabilization to coordinate sequence"""
        # Filter out None entries (skipped frames)
        valid_indices = [i for i, c in enumerate(coords) if c is not None]

        if len(valid_indices) <= 1:
            return coords  # Nothing to stabilize

        if mode == "smooth":
            # Gaussian smoothing
            for key in ['x', 'y', 'w', 'h']:
                values = [coords[i][key] for i in valid_indices]
                smoothed = gaussian_smooth_1d(values, window_size)
                for idx, i in enumerate(valid_indices):
                    coords[i][key] = int(round(smoothed[idx]))

        elif mode == "lock_first":
            # Use first valid frame's size
            first = coords[valid_indices[0]]
            ref_w, ref_h = first['w'], first['h']
            for i in valid_indices[1:]:
                # Adjust x, y to center the mask in ref_w x ref_h
                center_x = coords[i]['x'] + coords[i]['w'] // 2
                center_y = coords[i]['y'] + coords[i]['h'] // 2
                coords[i]['w'] = ref_w
                coords[i]['h'] = ref_h
                coords[i]['x'] = center_x - ref_w // 2
                coords[i]['y'] = center_y - ref_h // 2

        elif mode == "lock_largest":
            # Use max dimensions across all frames
            max_w = max(coords[i]['w'] for i in valid_indices)
            max_h = max(coords[i]['h'] for i in valid_indices)
            for i in valid_indices:
                center_x = coords[i]['x'] + coords[i]['w'] // 2
                center_y = coords[i]['y'] + coords[i]['h'] // 2
                coords[i]['w'] = max_w
                coords[i]['h'] = max_h
                coords[i]['x'] = center_x - max_w // 2
                coords[i]['y'] = center_y - max_h // 2

        elif mode == "median":
            # Median filter each coordinate
            for key in ['x', 'y', 'w', 'h']:
                values = [coords[i][key] for i in valid_indices]
                filtered = median_filter_1d(values, window_size)
                for idx, i in enumerate(valid_indices):
                    coords[i][key] = int(round(filtered[idx]))

        print(f"[DEBUG] Stabilization applied: mode={mode}, window={window_size}, valid_frames={len(valid_indices)}")
        return coords

    def preprocess_mask_for_stabilization(self, mask, mask_fill_holes, mask_expand_pixels,
            mask_invert, mask_blend_pixels, mask_hipass_filter):
        """Run mask preprocessing to extract coordinates for stabilization.
        Replicates the same preprocessing steps as inpaint_crop_single_image.
        Returns processed mask without modifying the original."""
        m = mask.clone()

        if mask_fill_holes:
            m = fillholes_iterative_hipass_fill_m(m)
        if mask_expand_pixels > 0:
            m = expand_m(m, mask_expand_pixels)
        if mask_invert:
            m = invert_m(m)
        if mask_blend_pixels > 0:
            m = expand_m(m, mask_blend_pixels)
            m = blur_m(m, mask_blend_pixels * 0.5)
        if mask_hipass_filter >= 0.01:
            m = hipassfilter_m(m, mask_hipass_filter)

        return m

    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask=None, optional_context_mask=None, stabilize_crop=False, stabilization_mode="smooth", smooth_window=5):

        output_padding = int(output_padding)
        
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch input')
            print(image.shape, type(image), image.dtype)
            if mask is not None:
                print(mask.shape, type(mask), mask.dtype)
            if optional_context_mask is not None:
                print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

        if image.shape[0] > 1:
            assert output_resize_to_target_size, "output_resize_to_target_size must be enabled when input is a batch of images, given all images in the batch output have to be the same size"

        # When a LoadImage node passes a mask without user editing, it may be the wrong shape.
        # Detect and fix that to avoid shape mismatch errors.
        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        # If no mask is provided, create one with the shape of the image
        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
    
        # If there is only one image for many masks, replicate it for all masks
        if mask.shape[0] > 1 and image.shape[0] == 1:
            assert image.dim() == 4, f"Expected 4D BHWC image tensor, got {image.shape}"
            image = image.expand(mask.shape[0], -1, -1, -1).clone()

        # If there is only one mask for many images, replicate it for all images
        if image.shape[0] > 1 and mask.shape[0] == 1:
            assert mask.dim() == 3, f"Expected 3D BHW mask tensor, got {mask.shape}"
            mask = mask.expand(image.shape[0], -1, -1).clone()

        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])

        # If there is only one optional_context_mask for many images, replicate it for all images
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            assert optional_context_mask.dim() == 3, f"Expected 3D BHW optional_context_mask tensor, got {optional_context_mask.shape}"
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch ready')
            print(image.shape, type(image), image.dtype)
            print(mask.shape, type(mask), mask.dtype)
            print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

         # Validate data
        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"
        assert optional_context_mask.ndimension() == 3, f"Expected 3 dimensions for optional_context_mask, got {optional_context_mask.ndimension()}"
        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert optional_context_mask.shape[1:] == image.shape[1:3], f"optional_context_mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {optional_context_mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"
        assert optional_context_mask.shape[0] == image.shape[0], f"Optional context mask batch does not match image batch. Expected {image.shape[0]}, got {optional_context_mask.shape[0]}"

        # Run for each image separately
        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
            # Frame skipping fields
            'skipped_indices': [],      # List of frame indices that were skipped (no mask)
            'original_frames': [],      # Original frames for skipped indices
            'total_frames': 0,          # Total input batch size
        }
        
        result_image = []
        result_mask = []

        debug_outputs = {name: [] for name in self.RETURN_NAMES if name.startswith("DEBUG_")}

        # Print GPU info at the start
        print_gpu_info()

        batch_size = image.shape[0]
        print(f"\n[DEBUG] Starting batch processing: {batch_size} image(s)")
        print(f"[DEBUG] Image shape: {image.shape}, device: {image.device}")
        batch_start_time = time.time()

        # Stabilization: Two-pass processing when enabled
        stabilized_coords = None
        if stabilize_crop and batch_size > 1:
            print(f"\n[DEBUG] ========== Stabilization Pass ==========")
            print(f"[DEBUG] Analyzing {batch_size} frames for stabilization (mode={stabilization_mode}, window={smooth_window})...")
            device = comfy.model_management.get_torch_device()

            # First pass: Preprocess masks and extract coordinates
            raw_coords = []
            for b in range(batch_size):
                one_mask_stab = mask[b].unsqueeze(0).to(device)

                mask_sum = torch.count_nonzero(one_mask_stab)
                if mask_sum == 0:
                    raw_coords.append(None)  # Will be skipped
                    print(f"[DEBUG] Frame {b}: Empty mask (will skip)")
                else:
                    # Preprocess mask (same steps as in inpaint_crop_single_image)
                    processed = self.preprocess_mask_for_stabilization(
                        one_mask_stab, mask_fill_holes, mask_expand_pixels,
                        mask_invert, mask_blend_pixels, mask_hipass_filter)
                    _, x, y, w, h = findcontextarea_m(processed)
                    raw_coords.append({'x': x, 'y': y, 'w': w, 'h': h})
                    print(f"[DEBUG] Frame {b}: Raw coords ({x}, {y}, {w}x{h})")

            # Apply stabilization
            stabilized_coords = self.stabilize_coordinates(raw_coords, stabilization_mode, smooth_window)

            # Print stabilized coords for comparison
            for b, coord in enumerate(stabilized_coords):
                if coord is not None:
                    print(f"[DEBUG] Frame {b}: Stabilized coords ({coord['x']}, {coord['y']}, {coord['w']}x{coord['h']})")

            print(f"[DEBUG] ========== Stabilization Pass Complete ==========\n")

        for b in range(batch_size):
            print(f"\n[DEBUG] ========== Processing image {b+1}/{batch_size} ==========")
            image_start_time = time.time()

            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = optional_context_mask[b].unsqueeze(0)

            # Check if mask is empty BEFORE processing - skip frames without masks
            mask_sum = torch.count_nonzero(one_mask)
            if mask_sum == 0:
                print(f"[DEBUG] Frame {b}: No mask detected - SKIPPING (will use original in output)")
                result_stitcher['skipped_indices'].append(b)
                result_stitcher['original_frames'].append(one_image.squeeze(0))
                continue  # Skip this frame entirely

            # Get stabilized coords for this frame (if stabilization enabled)
            frame_stabilized_coords = None
            if stabilized_coords is not None and b < len(stabilized_coords):
                frame_stabilized_coords = stabilized_coords[b]

            outputs = self.inpaint_crop_single_image(
                one_image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode,
                preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height,
                extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels,
                context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height,
                output_padding, one_mask, one_optional_context_mask, frame_stabilized_coords)

            stitcher, cropped_image, cropped_mask = outputs[:3]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                result_stitcher[key].append(stitcher[key])

            cropped_image = cropped_image.squeeze(0)
            result_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            result_mask.append(cropped_mask)

            # Handle the DEBUG_ fields dynamically
            for name, output in zip(self.RETURN_NAMES[3:], outputs[3:]):  # Start from index 3 since first 3 are fixed
                if name.startswith("DEBUG_"):
                    output_array = output.squeeze(0)  # Assuming output needs to be squeezed similar to image/mask
                    debug_outputs[name].append(output_array)

            image_time = time.time() - image_start_time
            print(f"[DEBUG] Image {b+1}/{batch_size} completed in {image_time:.4f}s")

        # Store total frames for reconstruction in stitch
        result_stitcher['total_frames'] = batch_size

        # Handle edge case: ALL frames were skipped (no masks in any frame)
        if len(result_image) == 0:
            print(f"[DEBUG] WARNING: All {batch_size} frames skipped (no masks detected in any frame)")
            print(f"[DEBUG] Returning original images unchanged")
            # Return original images with empty masks - stitch will pass them through
            result_image = image
            result_mask = torch.zeros((batch_size, image.shape[1], image.shape[2]), device=image.device)
        else:
            result_image = torch.stack(result_image, dim=0)
            result_mask = torch.stack(result_mask, dim=0)
            print(f"[DEBUG] Processed {len(result_image)} frames, skipped {len(result_stitcher['skipped_indices'])} frames")

        batch_time = time.time() - batch_start_time
        print(f"\n[DEBUG] ========== Batch processing complete ==========")
        print(f"[DEBUG] Total batch time: {batch_time:.4f}s")
        print(f"[DEBUG] Average time per image: {batch_time/batch_size:.4f}s\n")

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch output')
            print(result_image.shape, type(result_image), result_image.dtype)
            print(result_mask.shape, type(result_mask), result_mask.dtype)

        debug_outputs = {name: torch.stack(values, dim=0) for name, values in debug_outputs.items()}

        return result_stitcher, result_image, result_mask, *[debug_outputs[name] for name in self.RETURN_NAMES if name.startswith("DEBUG_")]


    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask, optional_context_mask, stabilized_coords=None):
        # Move tensors to GPU for processing
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        mask = mask.to(device)
        optional_context_mask = optional_context_mask.to(device)
        print(f"[DEBUG] inpaint_crop_single_image: Input size {image.shape[2]}x{image.shape[1]}, device={image.device}")

        if preresize:
            print(f"[DEBUG] Step 1: Pre-resizing image...")
            step_start = time.time()
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
            print(f"[DEBUG] Step 1 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
       
        if mask_fill_holes:
            print(f"[DEBUG] Step 2: Filling holes in mask...")
            step_start = time.time()
            mask = fillholes_iterative_hipass_fill_m(mask)
            print(f"[DEBUG] Step 2 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_fillholes_mask = mask.clone()

        if mask_expand_pixels > 0:
            print(f"[DEBUG] Step 3: Expanding mask by {mask_expand_pixels} pixels...")
            step_start = time.time()
            mask = expand_m(mask, mask_expand_pixels)
            print(f"[DEBUG] Step 3 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_expand_mask = mask.clone()

        if mask_invert:
            print(f"[DEBUG] Step 4: Inverting mask...")
            mask = invert_m(mask)
        if self.DEBUG_MODE:
            DEBUG_invert_mask = mask.clone()

        if mask_blend_pixels > 0:
            print(f"[DEBUG] Step 5: Creating blend mask (expand + blur)...")
            step_start = time.time()
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
            print(f"[DEBUG] Step 5 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_blur_mask = mask.clone()

        if mask_hipass_filter >= 0.01:
            print(f"[DEBUG] Step 6: Applying hipass filter...")
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if self.DEBUG_MODE:
            DEBUG_hipassfilter_mask = mask.clone()

        if extend_for_outpainting:
            print(f"[DEBUG] Step 7: Extending image for outpainting...")
            step_start = time.time()
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
            print(f"[DEBUG] Step 7 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()

        print(f"[DEBUG] Step 8: Finding context area from mask...")
        if stabilized_coords is not None:
            # Use pre-computed stabilized coordinates
            x, y = stabilized_coords['x'], stabilized_coords['y']
            w, h = stabilized_coords['w'], stabilized_coords['h']

            # Bounds checking to prevent out-of-bounds access
            img_h, img_w = mask.shape[1], mask.shape[2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            context = mask[:, y:y+h, x:x+w]
            print(f"[DEBUG] Step 8: Using STABILIZED context area at ({x}, {y}) with size {w}x{h}")
        else:
            context, x, y, w, h = findcontextarea_m(mask)
            # If no mask, mask everything for some inpainting.
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, image.shape[2], image.shape[1]
                context = mask[:, y:y+h, x:x+w]
            print(f"[DEBUG] Step 8: Found context area at ({x}, {y}) with size {w}x{h}")
        if self.DEBUG_MODE:
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if context_from_mask_extend_factor >= 1.01:
            print(f"[DEBUG] Step 9: Growing context area by factor {context_from_mask_extend_factor}...")
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
            print(f"[DEBUG] Step 9: New context area at ({x}, {y}) with size {w}x{h}")
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)

        print(f"[DEBUG] Step 10: Combining with optional context mask...")
        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        print(f"[DEBUG] Step 10: Final context area at ({x}, {y}) with size {w}x{h}")
        if self.DEBUG_MODE:
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        print(f"[DEBUG] Step 11: Performing magic crop and resize...")
        step_start = time.time()
        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        print(f"[DEBUG] Step 11 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
            print(f"[DEBUG] Step 12: Creating final blend mask...")
            step_start = time.time()
            cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
            print(f"[DEBUG] Step 12 completed in {time.time() - step_start:.4f}s")
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        print(f"[DEBUG] Single image processing complete. Output size: {cropped_image.shape[2]}x{cropped_image.shape[1]}")

        # Move results back to intermediate device (CPU by default)
        intermediate = comfy.model_management.intermediate_device()
        cropped_image = cropped_image.to(intermediate)
        cropped_mask = cropped_mask.to(intermediate)
        canvas_image = canvas_image.to(intermediate)
        cropped_mask_blend = cropped_mask_blend.to(intermediate)

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend




class InpaintStitchImproved:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "inpaint"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"


    def inpaint_stitch(self, stitcher, inpainted_image):
        results = []

        # Get frame skipping info
        skipped_indices = set(stitcher.get('skipped_indices', []))
        original_frames = stitcher.get('original_frames', [])
        total_frames = stitcher.get('total_frames', inpainted_image.shape[0])

        # If no frames were skipped, use original logic
        if len(skipped_indices) == 0:
            batch_size = inpainted_image.shape[0]
            assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, "Stitch batch size doesn't match image batch size"
            override = False
            if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
                override = True
            for b in range(batch_size):
                one_image = inpainted_image[b]
                one_stitcher = {}
                for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                    one_stitcher[key] = stitcher[key]
                for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                    if override: # One stitcher for many images, always read 0.
                        one_stitcher[key] = stitcher[key][0]
                    else:
                        one_stitcher[key] = stitcher[key][b]
                one_image = one_image.unsqueeze(0)
                one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
                one_image = one_image.squeeze(0)
                results.append(one_image)
        else:
            # Reconstruct full batch with skipped frames inserted
            print(f"[DEBUG] Stitch: Reconstructing {total_frames} frames ({len(skipped_indices)} skipped, {inpainted_image.shape[0]} inpainted)")
            inpainted_idx = 0
            original_idx = 0

            for frame_idx in range(total_frames):
                if frame_idx in skipped_indices:
                    # Use original frame (no inpainting needed)
                    print(f"[DEBUG] Frame {frame_idx}: Using original (was skipped)")
                    results.append(original_frames[original_idx])
                    original_idx += 1
                else:
                    # Stitch inpainted frame
                    one_image = inpainted_image[inpainted_idx]
                    one_stitcher = {}
                    for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                        one_stitcher[key] = stitcher[key]
                    for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                        one_stitcher[key] = stitcher[key][inpainted_idx]
                    one_image = one_image.unsqueeze(0)
                    one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
                    one_image = one_image.squeeze(0)
                    results.append(one_image)
                    inpainted_idx += 1

        result_batch = torch.stack(results, dim=0)

        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        # Move tensors to GPU for processing
        device = comfy.model_management.get_torch_device()

        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image'].to(device)

        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']

        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']

        mask = stitcher['cropped_mask_for_blend'].to(device)  # shape: [1, H, W]
        inpainted_image = inpainted_image.to(device)

        output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)

        # Move result back to intermediate device (CPU by default)
        output_image = output_image.to(comfy.model_management.intermediate_device())

        return (output_image,)


class StitcherDebugInfo:
    """
    Debug node that displays stitcher metadata as JSON text.
    Useful for debugging and understanding what the crop node captured.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "inpaint"
    DESCRIPTION = "Display stitcher metadata (frame counts, coordinates, skipped frames)"

    def preview(self, stitcher):
        import json

        batch_size = len(stitcher.get('canvas_to_orig_x', []))
        total_frames = stitcher.get('total_frames', batch_size)
        skipped_indices = stitcher.get('skipped_indices', [])

        info = {
            "total_input_frames": total_frames,
            "processed_frames": batch_size,
            "skipped_frames_count": len(skipped_indices),
            "skipped_frame_indices": skipped_indices,
            "algorithms": {
                "downscale": stitcher.get('downscale_algorithm'),
                "upscale": stitcher.get('upscale_algorithm'),
            },
            "blend_pixels": stitcher.get('blend_pixels'),
            "frames": []
        }

        for i in range(batch_size):
            canvas_img = stitcher['canvas_image'][i]
            canvas_shape = list(canvas_img.shape) if hasattr(canvas_img, 'shape') else "N/A"

            frame_info = {
                "frame_index": i,
                "canvas_to_orig": {
                    "x": stitcher['canvas_to_orig_x'][i],
                    "y": stitcher['canvas_to_orig_y'][i],
                    "w": stitcher['canvas_to_orig_w'][i],
                    "h": stitcher['canvas_to_orig_h'][i],
                },
                "crop_to_canvas": {
                    "x": stitcher['cropped_to_canvas_x'][i],
                    "y": stitcher['cropped_to_canvas_y'][i],
                    "w": stitcher['cropped_to_canvas_w'][i],
                    "h": stitcher['cropped_to_canvas_h'][i],
                },
                "canvas_shape": canvas_shape,
            }
            info["frames"].append(frame_info)

        text = json.dumps(info, indent=2)

        return {"ui": {"text": (text,)}}


class StitcherDebugImages:
    """
    Extract canvas images and masks from stitcher for workflow use and visualization.
    Returns canvas images, mask overlay visualization, and raw masks.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
            },
            "optional": {
                "mask_overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "mask_color": (["red", "green", "blue", "yellow", "magenta", "cyan"], {"default": "red"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("canvas_images", "canvas_with_mask_overlay", "masks")
    FUNCTION = "get_images"
    CATEGORY = "inpaint"
    DESCRIPTION = "Extract canvas images and masks from stitcher with optional mask visualization overlay"

    def get_images(self, stitcher, mask_overlay_opacity=0.5, mask_color="red"):
        canvas_images = stitcher.get('canvas_image', [])
        blend_masks = stitcher.get('cropped_mask_for_blend', [])

        if not canvas_images or len(canvas_images) == 0:
            empty_img = torch.zeros(1, 1, 1, 3)
            empty_mask = torch.zeros(1, 1, 1)
            return (empty_img, empty_img, empty_mask)

        # Process canvas images - squeeze extra batch dimensions
        images = []
        for canvas_img in canvas_images:
            img = canvas_img
            while img.dim() > 3:
                img = img.squeeze(0)
            images.append(img)

        canvas_batch = torch.stack(images, dim=0)  # [B, H, W, C]

        # Process masks and place them on canvas-sized masks using crop coordinates
        # The cropped_mask_for_blend is at crop resolution, need to place it on canvas
        canvas_masks = []
        for i, mask in enumerate(blend_masks):
            m = mask
            while m.dim() > 2:
                m = m.squeeze(0)

            # Get canvas dimensions from the processed canvas image
            canvas_h, canvas_w = images[i].shape[0], images[i].shape[1]

            # Get placement coordinates (where crop goes on canvas)
            ctc_x = stitcher['cropped_to_canvas_x'][i]
            ctc_y = stitcher['cropped_to_canvas_y'][i]
            ctc_w = stitcher['cropped_to_canvas_w'][i]
            ctc_h = stitcher['cropped_to_canvas_h'][i]

            # Create full canvas-sized mask
            full_mask = torch.zeros(canvas_h, canvas_w, device=m.device, dtype=m.dtype)

            # Resize cropped mask to target size and place on canvas
            m_resized = torch.nn.functional.interpolate(
                m.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                size=(ctc_h, ctc_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # back to [H, W]

            # Place resized mask on canvas at correct position
            full_mask[ctc_y:ctc_y+ctc_h, ctc_x:ctc_x+ctc_w] = m_resized

            canvas_masks.append(full_mask)

        mask_batch = torch.stack(canvas_masks, dim=0)  # [B, H, W]

        # Create overlay visualization (canvas with colored mask area)
        color_map = {
            "red": [1.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
            "yellow": [1.0, 1.0, 0.0],
            "magenta": [1.0, 0.0, 1.0],
            "cyan": [0.0, 1.0, 1.0],
        }
        color = torch.tensor(color_map[mask_color], device=canvas_batch.device)

        overlay_batch = canvas_batch.clone()
        mask_expanded = mask_batch.unsqueeze(-1).to(canvas_batch.device)  # [B, H, W, 1]
        color_layer = color.view(1, 1, 1, 3).expand_as(overlay_batch)

        # Blend: original * (1 - mask*opacity) + color * mask * opacity
        overlay_batch = overlay_batch * (1 - mask_expanded * mask_overlay_opacity) + \
                        color_layer * mask_expanded * mask_overlay_opacity

        return (canvas_batch, overlay_batch, mask_batch)


def expand_stitcher_metadata(stitcher, factor, interpolated_indices):
    """Expand stitcher lists to match new frame count after temporal expansion"""
    expanded = {
        'downscale_algorithm': stitcher['downscale_algorithm'],
        'upscale_algorithm': stitcher['upscale_algorithm'],
        'blend_pixels': stitcher['blend_pixels'],
        'skipped_indices': [],
        'original_frames': [],
        'total_frames': 0,
        'interpolated_indices': interpolated_indices,  # Track which frames are interpolated
        'original_frame_count': len(stitcher['canvas_to_orig_x']),  # Original count before expansion
    }

    # Expand per-frame lists
    list_keys = ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h',
                 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y',
                 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']

    for key in list_keys:
        expanded[key] = []
        original_list = stitcher[key]

        for i in range(len(original_list) - 1):
            # Add original frame data
            expanded[key].append(original_list[i])
            # Add interpolated frame copies (use data from frame i for stitching)
            for _ in range(factor - 1):
                if key == 'canvas_image':
                    # For canvas images, use the original (interpolation happens on cropped)
                    expanded[key].append(original_list[i])
                elif key == 'cropped_mask_for_blend':
                    # For blend masks, use the original
                    expanded[key].append(original_list[i])
                else:
                    # Numeric values - copy from previous frame
                    expanded[key].append(original_list[i])

        # Add final frame
        expanded[key].append(original_list[-1])

    expanded['total_frames'] = len(expanded['canvas_to_orig_x'])
    return expanded


def collapse_stitcher_metadata(stitcher, keep_indices):
    """Collapse stitcher lists back to original frame count"""
    collapsed = {
        'downscale_algorithm': stitcher['downscale_algorithm'],
        'upscale_algorithm': stitcher['upscale_algorithm'],
        'blend_pixels': stitcher['blend_pixels'],
        'skipped_indices': stitcher.get('skipped_indices', []),
        'original_frames': stitcher.get('original_frames', []),
        'total_frames': len(keep_indices),
    }

    list_keys = ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h',
                 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y',
                 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']

    for key in list_keys:
        original_list = stitcher[key]
        collapsed[key] = [original_list[i] for i in keep_indices]

    return collapsed


class TemporalExpand:
    """
    Expands frame count by inserting interpolated frames between originals.
    Works with STITCHER to maintain crop/stitch compatibility.
    Use this to "slow down" motion before inpainting for better results.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                "expansion_factor": (["2x", "3x", "4x"], {"default": "2x", "tooltip": "How many frames to create. 2x inserts 1 frame between each pair."}),
                "interpolation_mode": (["linear", "nearest"], {"default": "linear", "tooltip": "linear=blend frames, nearest=duplicate frames"}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("expanded_stitcher", "expanded_images", "expanded_masks")
    FUNCTION = "expand"
    CATEGORY = "inpaint"
    DESCRIPTION = "Insert interpolated frames between originals for smoother inpainting"

    def expand(self, stitcher, images, masks, expansion_factor="2x", interpolation_mode="linear"):
        factor = {"2x": 2, "3x": 3, "4x": 4}[expansion_factor]
        batch_size = images.shape[0]

        print(f"[DEBUG] TemporalExpand: {batch_size} frames → {expansion_factor} expansion ({interpolation_mode} mode)")

        if batch_size < 2:
            # Single frame, nothing to expand
            print(f"[DEBUG] TemporalExpand: Only 1 frame, skipping expansion")
            return (stitcher, images, masks)

        expanded_images = []
        expanded_masks = []
        interpolated_indices = []  # Track which frames are interpolated

        for i in range(batch_size - 1):
            frame_a = images[i]
            frame_b = images[i + 1]
            mask_a = masks[i]
            mask_b = masks[i + 1]

            # Add original frame
            expanded_images.append(frame_a)
            expanded_masks.append(mask_a)

            # Add interpolated frames
            for j in range(1, factor):
                alpha = j / factor
                if interpolation_mode == "linear":
                    interp_frame = frame_a * (1 - alpha) + frame_b * alpha
                    interp_mask = mask_a * (1 - alpha) + mask_b * alpha
                else:  # nearest
                    interp_frame = frame_a if alpha < 0.5 else frame_b
                    interp_mask = mask_a if alpha < 0.5 else mask_b

                expanded_images.append(interp_frame)
                expanded_masks.append(interp_mask)
                interpolated_indices.append(len(expanded_images) - 1)

        # Add final frame
        expanded_images.append(images[-1])
        expanded_masks.append(masks[-1])

        # Stack results
        expanded_images = torch.stack(expanded_images, dim=0)
        expanded_masks = torch.stack(expanded_masks, dim=0)

        # Expand stitcher metadata
        expanded_stitcher = expand_stitcher_metadata(stitcher, factor, interpolated_indices)

        print(f"[DEBUG] TemporalExpand: Expanded {batch_size} → {expanded_images.shape[0]} frames")
        print(f"[DEBUG] TemporalExpand: {len(interpolated_indices)} interpolated frames at indices: {interpolated_indices[:10]}{'...' if len(interpolated_indices) > 10 else ''}")

        return (expanded_stitcher, expanded_images, expanded_masks)


class TemporalCollapse:
    """
    Removes interpolated frames to restore original frame count.
    Uses indices from TemporalExpand to know which frames to keep.
    Use after inpainting to get back to original frame count.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE")
    RETURN_NAMES = ("collapsed_stitcher", "collapsed_images")
    FUNCTION = "collapse"
    CATEGORY = "inpaint"
    DESCRIPTION = "Remove interpolated frames to restore original frame count"

    def collapse(self, stitcher, images):
        interpolated_indices = set(stitcher.get('interpolated_indices', []))
        original_count = stitcher.get('original_frame_count', images.shape[0])

        print(f"[DEBUG] TemporalCollapse: {images.shape[0]} frames, {len(interpolated_indices)} interpolated")

        if not interpolated_indices:
            # No interpolation was done
            print(f"[DEBUG] TemporalCollapse: No interpolated frames found, returning unchanged")
            return (stitcher, images)

        # Keep only non-interpolated frames
        keep_indices = [i for i in range(images.shape[0]) if i not in interpolated_indices]
        collapsed_images = images[keep_indices]

        # Collapse stitcher metadata
        collapsed_stitcher = collapse_stitcher_metadata(stitcher, keep_indices)

        print(f"[DEBUG] TemporalCollapse: Collapsed {images.shape[0]} → {collapsed_images.shape[0]} frames")
        print(f"[DEBUG] TemporalCollapse: Kept indices: {keep_indices[:10]}{'...' if len(keep_indices) > 10 else ''}")

        return (collapsed_stitcher, collapsed_images)


class MaskRegionAnalyzer:
    """
    Analyzes mask video to detect and track individual regions (e.g., player silhouettes).
    Assigns persistent IDs to each region across frames using centroid-based tracking.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_region_size": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Minimum pixel count for a region to be tracked (filters noise)"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Binarization threshold for mask"}),
                "max_track_distance": ("INT", {"default": 150, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Maximum centroid movement (pixels) between frames for tracking"}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "INT", "INT")
    RETURN_NAMES = ("tracked_mask", "tracking_info", "total_players", "max_concurrent")
    FUNCTION = "analyze"
    CATEGORY = "inpaint"
    DESCRIPTION = "Detect and track mask regions across frames with persistent IDs"

    def analyze(self, mask, min_region_size, threshold, max_track_distance):
        print(f"[DEBUG] MaskRegionAnalyzer: Input mask shape {mask.shape}")

        tracked_mask, tracking_info = analyze_and_track_masks(
            mask,
            min_size=min_region_size,
            max_dist=max_track_distance,
            threshold=threshold
        )

        tracking_info_json = json.dumps(tracking_info, indent=2)
        total_players = tracking_info['total_players']
        max_concurrent = tracking_info['max_concurrent']

        print(f"[DEBUG] MaskRegionAnalyzer: Output - {total_players} total players, {max_concurrent} max concurrent")

        return (tracked_mask, tracking_info_json, total_players, max_concurrent)


class MaskPlayerFilter:
    """
    Filters a tracked mask to output only selected player IDs.
    Use batch_index to process players in groups (0=players 1-4, 1=players 5-8, etc.)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracked_mask": ("MASK",),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of players per batch"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Which batch to output (0=first N players, 1=next N, etc.)"}),
            },
            "optional": {
                "player_ids": ("STRING", {"default": "",
                    "tooltip": "Override: comma-separated player IDs (e.g., '1,2,3,4'). Leave empty to use batch_index."}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("filtered_mask", "selected_ids")
    FUNCTION = "filter_players"
    CATEGORY = "inpaint"
    DESCRIPTION = "Filter tracked mask to selected player batch"

    def filter_players(self, tracked_mask, batch_size, batch_index, player_ids=""):
        device = tracked_mask.device

        # Determine which player IDs to include
        if player_ids.strip():
            # Parse manual player IDs
            try:
                selected_ids = [int(x.strip()) for x in player_ids.split(",") if x.strip()]
            except ValueError:
                print(f"[DEBUG] MaskPlayerFilter: Invalid player_ids format, using batch_index")
                selected_ids = []
        else:
            selected_ids = []

        if not selected_ids:
            # Use batch_index to determine IDs
            start_id = batch_index * batch_size + 1
            end_id = start_id + batch_size
            selected_ids = list(range(start_id, end_id))

        print(f"[DEBUG] MaskPlayerFilter: Filtering for player IDs {selected_ids}")

        # Create binary mask where tracked_mask value is in selected_ids
        # tracked_mask has player IDs as float values (1.0, 2.0, 3.0, etc.)
        filtered_mask = torch.zeros_like(tracked_mask)

        for pid in selected_ids:
            # Match pixels with this player ID (with small tolerance for float comparison)
            player_pixels = (tracked_mask >= pid - 0.5) & (tracked_mask < pid + 0.5)
            filtered_mask[player_pixels] = 1.0

        selected_ids_str = ",".join(str(x) for x in selected_ids)
        print(f"[DEBUG] MaskPlayerFilter: Output mask has {(filtered_mask > 0).sum().item()} pixels")

        return (filtered_mask, selected_ids_str)


class MaskColorizer:
    """
    Debug visualization: colors each tracked player region with a unique color.
    Useful for verifying tracking quality.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracked_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("colored_image",)
    FUNCTION = "colorize"
    CATEGORY = "inpaint"
    DESCRIPTION = "Visualize tracked mask regions with unique colors"

    def colorize(self, tracked_mask):
        device = tracked_mask.device
        B, H, W = tracked_mask.shape

        # Define a distinct color palette (up to 20 colors)
        color_palette = torch.tensor([
            [0.0, 0.0, 0.0],       # 0: Background (black)
            [1.0, 0.0, 0.0],       # 1: Red
            [0.0, 1.0, 0.0],       # 2: Green
            [0.0, 0.0, 1.0],       # 3: Blue
            [1.0, 1.0, 0.0],       # 4: Yellow
            [1.0, 0.0, 1.0],       # 5: Magenta
            [0.0, 1.0, 1.0],       # 6: Cyan
            [1.0, 0.5, 0.0],       # 7: Orange
            [0.5, 0.0, 1.0],       # 8: Purple
            [0.0, 1.0, 0.5],       # 9: Spring Green
            [1.0, 0.5, 0.5],       # 10: Light Red
            [0.5, 1.0, 0.5],       # 11: Light Green
            [0.5, 0.5, 1.0],       # 12: Light Blue
            [0.8, 0.8, 0.0],       # 13: Olive
            [0.8, 0.0, 0.8],       # 14: Dark Magenta
            [0.0, 0.8, 0.8],       # 15: Teal
            [1.0, 0.8, 0.6],       # 16: Peach
            [0.6, 0.8, 1.0],       # 17: Sky Blue
            [0.8, 0.6, 1.0],       # 18: Lavender
            [0.6, 1.0, 0.8],       # 19: Mint
        ], device=device, dtype=torch.float32)

        # Create output image [B, H, W, 3]
        colored_image = torch.zeros(B, H, W, 3, device=device, dtype=torch.float32)

        for b in range(B):
            frame_mask = tracked_mask[b]  # [H, W]
            unique_ids = torch.unique(frame_mask)

            for uid in unique_ids:
                uid_int = int(uid.item())
                if uid_int == 0:
                    continue  # Skip background

                # Get color (cycle through palette if more players than colors)
                color_idx = uid_int % len(color_palette)
                if color_idx == 0:
                    color_idx = 1  # Avoid black for non-background

                color = color_palette[color_idx]

                # Apply color to pixels with this ID
                mask_pixels = (frame_mask >= uid_int - 0.5) & (frame_mask < uid_int + 0.5)
                colored_image[b, mask_pixels] = color

        print(f"[DEBUG] MaskColorizer: Created colored visualization {colored_image.shape}")

        return (colored_image,)


class MaskRegionLimiter:
    """
    Per-frame region limiter: selects up to N regions per frame based on size/position.
    Use multiple passes with different batch_index values to process all regions.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "max_regions": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Maximum regions to keep per frame"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Which batch of regions (0=first N, 1=next N, etc.)"}),
                "min_region_size": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "Minimum pixel count for a region to be considered"}),
                "selection_mode": (["largest", "smallest", "leftmost", "rightmost"], {"default": "largest",
                    "tooltip": "How to order regions for selection"}),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "STRING")
    RETURN_NAMES = ("filtered_mask", "max_batches_needed", "regions_info")
    FUNCTION = "limit_regions"
    CATEGORY = "inpaint"
    DESCRIPTION = "Limit mask to N regions per frame. Use batch_index to process remaining regions in additional passes."

    def limit_regions(self, mask, max_regions, batch_index, min_region_size, selection_mode):
        print(f"[DEBUG] MaskRegionLimiter: Input mask shape {mask.shape}")

        filtered_mask, max_batches, regions_per_frame = limit_regions_per_frame(
            mask,
            max_regions=max_regions,
            batch_index=batch_index,
            min_size=min_region_size,
            mode=selection_mode
        )

        # Build info string
        non_empty_frames = sum(1 for r in regions_per_frame if r > 0)
        info = {
            "batch_index": batch_index,
            "max_batches_needed": max_batches,
            "total_frames": len(regions_per_frame),
            "frames_with_regions": non_empty_frames,
            "max_regions_in_frame": max(regions_per_frame) if regions_per_frame else 0,
        }
        regions_info = json.dumps(info, indent=2)

        print(f"[DEBUG] MaskRegionLimiter: Output - batch {batch_index}/{max_batches}, {non_empty_frames} frames with regions")

        return (filtered_mask, max_batches, regions_info)
