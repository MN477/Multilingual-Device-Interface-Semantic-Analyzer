import os
import cv2
import numpy as np
import torch
from CRAFT.craft import CRAFT
from CRAFT.imgproc import resize_aspect_ratio
from CRAFT.craft_utils import getDetBoxes, adjustResultCoordinates

def strip_alpha_channel(img: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3 or img.shape[2] != 4:
        return img[:, :, :3] if len(img.shape) == 3 else img

    rgb = img[:, :, :3]
    alpha = img[:, :, 3]

    is_premultiplied = not np.any(rgb > alpha[:, :, np.newaxis])
    alpha_f = alpha.astype(np.float32)
    rgb_f = rgb.astype(np.float32)

    if is_premultiplied:
        mask = alpha > 0
        rgb_out = np.zeros_like(rgb_f)
        rgb_out[mask] = (rgb_f[mask] / alpha_f[mask][:, np.newaxis]) * 255.0
        return np.clip(rgb_out, 0, 255).astype(np.uint8)
    else:
        alpha_norm = alpha_f / 255.0
        alpha_norm = alpha_norm[:, :, np.newaxis]
        rgb_out = rgb_f * alpha_norm + 255.0 * (1.0 - alpha_norm)
        return np.clip(rgb_out, 0, 255).astype(np.uint8)


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_net = CRAFT().to(_DEVICE)
state_dict = torch.load("craft_mlt_25k.pth", map_location=_DEVICE)

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
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        img_rgb,
        canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio
    )

    ratio_h = ratio_w = 1 / target_ratio

    x = img_resized.astype(np.float32)
    x = (x - 127.5) / 127.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        y, _ = _net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    boxes, polys = getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly=True
    )

    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is not None:
            polys[k] = polys[k] * (ratio_w * 2, ratio_h * 2)

    final_polys = []
    for box, poly in zip(boxes, polys):
        if poly is None:
            final_polys.append(np.array(box, dtype=np.float32))
        else:
            final_polys.append(np.array(poly, dtype=np.float32))

    return final_polys


def localize_text(image_path: str, **craft_kwargs) -> tuple[np.ndarray, list[np.ndarray]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target file path context does not exist: {image_path}")

    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError(f"Unable to read file contents into a valid matrix format: {image_path}")

    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGBA)
    elif len(img_raw.shape) == 3 and img_raw.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)

    img_clean_rgb = strip_alpha_channel(img_rgb)
    raw_polygons = detect_text_regions(img_clean_rgb, **craft_kwargs)

    return img_clean_rgb, raw_polygons


def draw_polygons(img: np.ndarray, polygons: list[np.ndarray], output_path: str):
    debug_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for poly in polygons:
        if len(poly) == 0:
            continue
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        p1 = tuple(pts[0][0])
        cv2.circle(debug_img, p1, radius=4, color=(0, 0, 255), thickness=-1)
        
    cv2.imwrite(output_path, debug_img)
    print(f"Debug image saved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        try:
            print(f"Processing {test_img_path}...")
            img_rgb, polys = localize_text(test_img_path)
            print(f"Found {len(polys)} raw text regions.")
            
            debug_dir = "debug_img_text_local"
            os.makedirs(debug_dir, exist_ok=True)
            output_path = os.path.join(debug_dir, f"debug_{os.path.basename(test_img_path)}")
            draw_polygons(img_rgb, polys, output_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide an image path to test.")
