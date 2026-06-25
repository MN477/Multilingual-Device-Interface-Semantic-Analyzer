import os
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon as ShapelyPolygon
from CRAFT.craft import CRAFT
from CRAFT.imgproc import resize_aspect_ratio
from CRAFT.craft_utils import getDetBoxes, adjustResultCoordinates

def strip_alpha_channel(img: np.ndarray) -> np.ndarray:
    """
    Checks for a 4-channel image (RGBA), determines if alpha is straight or 
    premultiplied, handles normalization/compositing appropriately, and returns 
    a clean 3-channel RGB image.
    
    Args:
        img (np.ndarray): Input image array (can be 3 or 4 channels).
        
    Returns:
        np.ndarray: Normalized 3-channel RGB image.
    """
    # If the image does not have 4 channels, return it as-is
    if len(img.shape) != 3 or img.shape[2] != 4:
        return img[:, :, :3] if len(img.shape) == 3 else img

    rgb = img[:, :, :3]
    alpha = img[:, :, 3]

    # Detect premultiplied alpha: if no RGB value is greater than its alpha, it's premultiplied.
    # We expand dimensions of alpha to broadcast across the 3 channels.
    is_premultiplied = not np.any(rgb > alpha[:, :, np.newaxis])

    alpha_f = alpha.astype(np.float32)
    rgb_f = rgb.astype(np.float32)

    if is_premultiplied:
        # Un-premultiply: RGB = (RGB / alpha) * 255 where alpha > 0
        mask = alpha > 0
        rgb_out = np.zeros_like(rgb_f)
        # Prevent division by zero by mapping only where alpha > 0
        rgb_out[mask] = (rgb_f[mask] / alpha_f[mask][:, np.newaxis]) * 255.0
        return np.clip(rgb_out, 0, 255).astype(np.uint8)
    else:
        # Straight alpha: composite onto a clean white background
        # RGB_out = RGB * alpha + 255 * (1 - alpha), where alpha is mapped to [0, 1]
        alpha_norm = alpha_f / 255.0
        alpha_norm = alpha_norm[:, :, np.newaxis]  # Reshape to (H, W, 1) for broadcasting
        
        rgb_out = rgb_f * alpha_norm + 255.0 * (1.0 - alpha_norm)
        return np.clip(rgb_out, 0, 255).astype(np.uint8)


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_net = CRAFT().to(_DEVICE)
state_dict = torch.load("craft_mlt_25k.pth", map_location=_DEVICE)

# Remove the "module." prefix added by DataParallel during training
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

_net.load_state_dict(new_state_dict)
_net.eval()


def detect_text_regions(
    img_rgb: np.ndarray,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5
) -> list[np.ndarray]:
    """
    Detect text regions using official PyTorch CRAFT with correct scaling.*
    Initializes the CRAFT text detector with the specified parameters and fetches
    raw text region polygons mapped back to original image space coordinates.
    Args:
    img_rgb (np.ndarray): 3-channel RGB image array.
    text_threshold (float): High confidence threshold for text boxes.
    link_threshold (float): Affinity score threshold linking characters into words.
    low_text (float): Lower bound threshold for character detection region.
    canvas_size (int): Inference image scaling size.
    mag_ratio (float): Image magnification multiplier ratio.
    Returns:
    list[np.ndarray]: List of 4-point bounding polygons.
    """


    # Resize (CRAFT internal logic)
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        img_rgb,
        canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio
    )

    ratio_h = ratio_w = 1 / target_ratio

    # 🔹 Normalize image (IMPORTANT)
    x = img_resized.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    # 🔹 Forward pass
    with torch.no_grad():
        y, _ = _net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # 🔹 Get boxes
    boxes, polys = getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly=True
    )

    # 🔹 Rescale to ORIGINAL image (CRITICAL)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys contains arrays of varying lengths, so adjustResultCoordinates's np.array() crashes on modern numpy.
    # We do it manually:
    for k in range(len(polys)):
        if polys[k] is not None:
            polys[k] = polys[k] * (ratio_w * 2, ratio_h * 2)

    # Replace None polys with boxes
    final_polys = []
    for box, poly in zip(boxes, polys):
        if poly is None:
            final_polys.append(np.array(box, dtype=np.float32))
        else:
            final_polys.append(np.array(poly, dtype=np.float32))

    return final_polys


def _shoelace_area(pts: np.ndarray) -> float:
    """Helper method to compute polygon surface area using the Shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _shortest_side(pts: np.ndarray) -> float:
    """Helper method to compute the Euclidean distance of the absolute shortest side."""
    rolled = np.roll(pts, -1, axis=0)
    distances = np.sqrt(np.sum((pts - rolled) ** 2, axis=1))
    return float(np.min(distances))


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Sorts a 4-point polygon into a consistent clockwise ordering pattern:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left. OpenCV requires this 
    standard structure for geometric operations like getPerspectiveTransform.
    """
    pts = pts.reshape(4, 2)
    # Sort points sequentially by X coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Sort left-most coordinates by Y to extract Top-Left and Bottom-Left
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most[0], left_most[1]
    
    # Sort right-most coordinates by Y to extract Top-Right and Bottom-Right
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most[0], right_most[1]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


def validate_and_filter_polygons(
    polygons: list[np.ndarray], 
    img_shape: tuple, 
    min_area: int = 100, 
    min_side: int = 5
) -> list[np.ndarray]:
    """
    Filters raw coordinate polygons based on area thresholds, minimum length, 
    topological self-intersections, and guarantees appropriate clockwise sorting.
    
    Args:
        polygons (list[np.ndarray]): List of 4-point text localization candidate polygons.
        img_shape (tuple): Source dimension limits of the image.
        min_area (int): Rejection threshold for shapes below this surface area size.
        min_side (int): Rejection threshold for elements with tiny pixel dimensions.
        
    Returns:
        list[np.ndarray]: Structural verified bounding polygons.
    """
    filtered_polygons = []

    for poly in polygons:
        # Verify the shape is a standard 4-point quadrilateral
        if poly.shape[0] != 4:
            continue

        # 1. Reject if Area < min_area via Shoelace formula
        if _shoelace_area(poly) < min_area:
            continue

        # 2. Reject if Shortest Side < min_side pixels
        if _shortest_side(poly) < min_side:
            continue

        # 3. Check for topological self-intersection using Shapely's validation engine
        shapely_poly = ShapelyPolygon(poly)
        if not shapely_poly.is_valid:
            continue

        # 4. Enforce proper Clockwise point ordering layout structure
        ordered_poly = _order_points_clockwise(poly)
        filtered_polygons.append(ordered_poly)

    return filtered_polygons


def localize_text(image_path: str, **craft_kwargs) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Executes the total continuous Phase 0 Text Localization processing pipeline.
    
    Args:
        image_path (str): File location string pointing to the source target image.
        **craft_kwargs: Dynamic hyperparameter overrides passed directly to detection engine.
        
    Returns:
        tuple[np.ndarray, list[np.ndarray]]: Processed 3-channel RGB image array 
                                             and a list of safe validated text coordinates.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target file path context does not exist: {image_path}")

    # Read the image unmodified to capture Alpha variations (PNG, JPEG, WebP)
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError(f"Unable to read file contents into a valid matrix format: {image_path}")

    # Convert native OpenCV default BGR structures over to standard RGB space
    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGBA)
    elif len(img_raw.shape) == 3 and img_raw.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        # Handle fallback cases like single-channel grayscale elements seamlessly
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)

    # 1. Execute Alpha Processing Pipelines
    img_clean_rgb = strip_alpha_channel(img_rgb)

    # 2. Extract Regional CRAFT Coordinate Predictions
    raw_polygons = detect_text_regions(img_clean_rgb, **craft_kwargs)

    # 3. Enforce Strict Geometric Filtering rules
    valid_polygons = validate_and_filter_polygons(raw_polygons, img_clean_rgb.shape)

    return img_clean_rgb, valid_polygons


def draw_polygons(img: np.ndarray, polygons: list[np.ndarray], output_path: str):
    """
    Visual debugging tool: draws the valid polygons on the image and saves it.
    """
    # OpenCV expects BGR for saving
    debug_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for poly in polygons:
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw the first point in red to verify clockwise ordering
        p1 = tuple(pts[0][0])
        cv2.circle(debug_img, p1, radius=4, color=(0, 0, 255), thickness=-1)
        
    cv2.imwrite(output_path, debug_img)
    print(f"Debug image saved to {output_path}")


if __name__ == "__main__":
    # Test script usage
    import sys
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        try:
            print(f"Processing {test_img_path}...")
            img_rgb, polys = localize_text(test_img_path)
            print(f"Found {len(polys)} valid text regions.")
            
            # Save debug visualization
            debug_dir = "debug_img_text_local"
            os.makedirs(debug_dir, exist_ok=True)
            output_path = os.path.join(debug_dir, f"debug_{os.path.basename(test_img_path)}")
            draw_polygons(img_rgb, polys, output_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide an image path to test.")