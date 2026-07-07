import json
import os
import glob
import numpy as np

GT_PATH = 'project-8-at-2026-06-30-14-56-34008404.json'
ENGINES = {
    'PaddleOCR_With_PrePost': 'paddleocr_det_output',
    'PaddleOCR_Without_PrePost': 'paddleocr_det_output - Copie'
}
IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [xmin, ymin, xmax, ymax]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0, 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Avoid division by zero
    if box1_area + box2_area - inter_area == 0:
        return 0.0, 0.0

    iou = inter_area / float(box1_area + box2_area - inter_area)
    # Also return intersection over predicted area (used for fragmentation)
    io_pred = inter_area / float(box2_area) if box2_area > 0 else 0.0
    return iou, io_pred

def load_gt():
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gt_dict = {}
    for task in data:
        img_path = task.get("data", {}).get("image", "")
        filename = img_path.split("/")[-1].split("?d=")[-1].replace("input_images/", "")
        if "%" in filename: # handle url encoding if any
            import urllib.parse
            filename = urllib.parse.unquote(filename)
        if not filename:
            continue
        
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        
        results = annotations[0].get("result", [])
        rects = [r for r in results if r["type"] == "rectanglelabels"]
        
        gt_boxes_pct = []
        for r in rects:
            v = r["value"]
            # coordinates are percentages 0-100
            x_pct, y_pct, w_pct, h_pct = v["x"], v["y"], v["width"], v["height"]
            gt_boxes_pct.append([x_pct, y_pct, w_pct, h_pct])
            
        # Clean filename just in case (e.g. remove query strings)
        filename = filename.split("?")[0]
        gt_dict[filename] = gt_boxes_pct
    return gt_dict

def evaluate():
    gt_dict = load_gt()
    print(f"Loaded {len(gt_dict)} Ground Truth images.")

    final_results = {}

    for engine_name, output_dir in ENGINES.items():
        if not os.path.exists(output_dir):
            print(f"Warning: Directory {output_dir} not found. Skipping.")
            continue

        json_files = glob.glob(os.path.join(output_dir, "*.json"))
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_latency = 0.0
        total_images = 0
        
        all_confidences = []
        
        # Fragmentation metric: how many predicted boxes intersect significantly with a single GT box
        # Ratio > 1 indicates semantic splitting (e.g., words instead of sentences)
        total_gt_boxes_evaluated = 0
        total_fragmentation_count = 0

        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            
            # Extract original filename
            img_path = pred_data.get("image_path", "")
            orig_filename = os.path.basename(img_path)
            
            if orig_filename not in gt_dict:
                continue
            
            total_images += 1
            total_latency += pred_data.get("inference_time_sec", 0.0)
            
            width, height = pred_data.get("image_size", [1, 1])
            if width == 0 or height == 0:
                width, height = 1, 1
            
            # Get GT boxes in absolute coordinates
            gt_boxes = []
            for pct in gt_dict[orig_filename]:
                x_min = pct[0] / 100.0 * width
                y_min = pct[1] / 100.0 * height
                w = pct[2] / 100.0 * width
                h = pct[3] / 100.0 * height
                gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            
            # Get Pred boxes
            pred_boxes = []
            for det in pred_data.get("detections", []):
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    pred_boxes.append(bbox)
                
                # Try to grab confidence
                conf = det.get("det_score") or det.get("confidence") or det.get("score")
                if conf is not None:
                    all_confidences.append(float(conf))
                    
            # Evaluate using IoU
            matched_gt = set()
            matched_pred = set()
            
            for p_idx, p_box in enumerate(pred_boxes):
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou, _ = calculate_iou(g_box, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
                
                if best_iou > IOU_THRESHOLD:
                    matched_pred.add(p_idx)
                    matched_gt.add(best_g_idx)
            
            tp = len(matched_pred)
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - len(matched_gt)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Evaluate Fragmentation (Semantic Connectivity)
            # For each GT box, how many pred boxes fall inside it?
            for g_box in gt_boxes:
                intersect_count = 0
                for p_box in pred_boxes:
                    _, io_pred = calculate_iou(g_box, p_box)
                    # If more than 30% of the prediction box overlaps with the GT box, count it
                    if io_pred > 0.3:
                        intersect_count += 1
                total_fragmentation_count += max(1, intersect_count) if intersect_count > 0 else 0
                total_gt_boxes_evaluated += 1

        # Calculate final metrics for the engine
        if total_images == 0:
            print(f"{engine_name}: No matched images found.")
            continue
            
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_latency = total_latency / total_images
        avg_conf = np.mean(all_confidences) if all_confidences else 0.0
        frag_ratio = total_fragmentation_count / total_gt_boxes_evaluated if total_gt_boxes_evaluated > 0 else 0.0
        
        final_results[engine_name] = {
            "Images Evaluated": total_images,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Avg Latency (s)": round(avg_latency, 3),
            "Avg Confidence": round(avg_conf, 4),
            "Fragmentation Ratio": round(frag_ratio, 2)
        }

    # Print and save results
    print("\n--- Final Evaluation Results ---")
    for engine, res in final_results.items():
        print(f"\n{engine}:")
        for k, v in res.items():
            print(f"  {k}: {v}")

    with open("evaluation_report.json", "w", encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
        
    print("\nResults saved to evaluation_report.json")

if __name__ == "__main__":
    evaluate()
