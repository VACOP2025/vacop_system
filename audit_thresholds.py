import os
import glob
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

MODEL_PATH = r'runs/detect/runs/train/yolo26_traffic_m_optim/weights/best.pt'
VAL_IMG_PATH = r'output_dataset/images/val'
THRESHOLDS = np.arange(0.05, 0.65, 0.05)

CUSTOM_CLASSES = {
    0: 'Person',
    1: 'Car',
    2: 'Stop Sign',
    3: 'Traffic Light',
    4: 'Traffic Light',
    5: 'Traffic Light'
}

def calculate_iou_batch(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    b1 = np.array(boxes1)
    b2 = np.array(boxes2)
    
    xi1 = np.maximum(b1[:, None, 0], b2[:, 0])
    yi1 = np.maximum(b1[:, None, 1], b2[:, 1])
    xi2 = np.minimum(b1[:, None, 2], b2[:, 2])
    yi2 = np.minimum(b1[:, None, 3], b2[:, 3])
    
    inter_area = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    box1_area = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    box2_area = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    
    union_area = box1_area[:, None] + box2_area[:] - inter_area
    
    return np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area!=0)

def get_ground_truths(img_path, shape):
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
    gts = defaultdict(list)
    
    if not os.path.exists(label_path): return gts
    
    h, w = shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            if cls_id not in CUSTOM_CLASSES: continue
            
            cx, cy, bw, bh = parts[1:5]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            
            cat_name = CUSTOM_CLASSES[cls_id]
            gts[cat_name].append([x1, y1, x2, y2])
    return gts

def run_audit():
    print(f"⚖️ DÉMARRAGE DE L'AUDIT DE SEUIL (0.05 -> 0.60)")

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur: Modèle introuvable ici : {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    image_files = glob.glob(os.path.join(VAL_IMG_PATH, "*.jpg"))
    
    if not image_files:
        print(f"Erreur: Aucune image trouvée dans {VAL_IMG_PATH}")
        return

    print(f"Images à traiter : {len(image_files)}")

    history = defaultdict(lambda: defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0}))

    for img_path in tqdm(image_files, desc="Processing Images"):
        img = cv2.imread(img_path)
        if img is None: continue
        
        gts = get_ground_truths(img_path, img.shape) 

        try:
            res = model.predict(img_path, conf=0.01, verbose=False)[0]
        except Exception as e:
            continue

        all_preds = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CUSTOM_CLASSES: continue
            
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            all_preds.append({'cls': CUSTOM_CLASSES[cls_id], 'conf': conf, 'box': xyxy})

        for thresh in THRESHOLDS:

            thresh_preds = defaultdict(list)
            for p in all_preds:
                if p['conf'] >= thresh:
                    thresh_preds[p['cls']].append(p['box'])

            for cat in ['Person', 'Car']:
                gt_boxes = gts[cat]
                pred_boxes = thresh_preds[cat]
                

                if not gt_boxes and not pred_boxes: continue
                
                if not pred_boxes:
                    history[thresh][cat]['FN'] += len(gt_boxes)
                    continue
                if not gt_boxes:
                    history[thresh][cat]['FP'] += len(pred_boxes)
                    continue

                iou_mat = calculate_iou_batch(pred_boxes, gt_boxes)
                matched_gt = set()
                tp = 0
                fp = 0
                
                for i in range(len(pred_boxes)):
                    if iou_mat.shape[1] == 0: break 
                    best_gt_idx = np.argmax(iou_mat[i])

                    if iou_mat[i][best_gt_idx] >= 0.5 and best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                
                fn = len(gt_boxes) - len(matched_gt)
                
                history[thresh][cat]['TP'] += tp
                history[thresh][cat]['FP'] += fp
                history[thresh][cat]['FN'] += fn

    print("\n" + "="*80)
    print(f"RÉSULTATS DE L'AUDIT DE SEUIL (Optimisation Rappel)")
    print("="*80)
    
    for cat in ['Person', 'Car']:
        print(f"\n🔹 CLASSE : {cat.upper()}")
        print(f"{'SEUIL':<10} | {'RAPPEL (Recall)':<18} | {'PRÉCISION':<18} | {'F1-SCORE':<10}")
        print("-" * 65)
        
        best_f1 = 0
        best_thresh = 0.25
        
        for thresh in THRESHOLDS:
            stats = history[thresh][cat]
            tp = stats['TP']
            fp = stats['FP']
            fn = stats['FN']
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            mark = ""
            if rec > 0.80: mark = "✅" 
            if rec > 0.90: mark = "🌟"
            
            print(f"{thresh:.2f}       | {rec:.1%} {mark:<5}       | {prec:.1%}            | {f1:.1%}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"\n💡 Recommandation pour {cat}: Seuil approx {best_thresh:.2f} (F1 Max)")
        print("-" * 65)

if __name__ == "__main__":
    run_audit()