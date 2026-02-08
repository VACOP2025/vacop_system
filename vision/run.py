""" Fichier principal pour exécuter les modèles """

import cv2
import torch
import numpy as np
import time
import threading
import queue
import tensorrt as trt
from ultralytics import YOLO
import torchvision.transforms as T
import gc
import subprocess



# --- CONFIGURATION ---

CAMERA_INDEX = 0  # "/dev/video0"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

YOLO_ENGINE = "../models/yolo26s.engine"
TWINLITE_ENGINE = "../models/twinlite.engine"
CLASSIFIER_ENGINE = "../models/classifier.engine"

CONF_THRESH = 0.25
CLS_CONF_THRESH = 0.85
YOLO_IMG_SIZE = 320

COLORS = {
    'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Yellow': (0, 255, 255),
    'Car': (100, 100, 100), 'Person': (255, 100, 0), 'Stop': (0, 0, 128)
}

class TRTModuleStatic:
    def __init__(self, engine_path, device):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            
            if -1 in shape:
                 real_shape = list(shape)
                 for dim_idx, dim in enumerate(real_shape):
                     if dim == -1: real_shape[dim_idx] = 16 
                 tensor = torch.zeros(tuple(real_shape), device=device, dtype=torch.float32)
            else:
                 tensor = torch.zeros(tuple(shape), device=device, dtype=torch.float32)

            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'tensor': tensor})
            else:
                self.outputs.append({'name': name, 'tensor': tensor})

    def forward(self, x, stream):
        input_info = self.inputs[0]
        if x.shape != input_info['tensor'].shape:
            self.context.set_input_shape(input_info['name'], x.shape)
        
        self.context.set_tensor_address(input_info['name'], x.data_ptr())
        self.context.execute_async_v3(stream.cuda_stream)
        return [out['tensor'] for out in self.outputs]


class USBCameraReader:
    """Lecteur de caméra USB optimisé avec thread séparé pour la capture."""
    
    def __init__(self, camera_index=0, width=1280, height=720, fps=30, queue_size=2):
        self.stopped = False
        self.q = queue.Queue(maxsize=queue_size)
        
        # Ouvrir la caméra avec le backend V4L2 (optimisé pour Linux/Jetson)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        # Configuration de la caméra
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Optionnel : utiliser MJPG pour de meilleures performances
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Réduire le buffer interne de la caméra pour minimiser la latence
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Récupérer les dimensions réelles
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f" Caméra ouverte: {self.w}x{self.h} @ {actual_fps} FPS")
        
        # Démarrer le thread de capture
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        """Thread de capture en continu."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print(" Erreur de lecture caméra")
                time.sleep(0.01)
                continue
            
            # Si la queue est pleine, on jette l'ancienne frame (toujours garder la plus récente)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            self.q.put(frame)
        
        self.cap.release()

    def read(self):
        """Récupère la dernière frame disponible."""
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def more(self):
        """Retourne True tant que la caméra fonctionne."""
        return not self.stopped

    def stop(self):
        """Arrête proprement la capture."""
        self.stopped = True
        self.t.join(timeout=2.0)
        print(" Caméra fermée")

# --- UTILS DESSIN ---
def draw_labeled_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, 0, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), 0, 0.6, (255, 255, 255), 1)

def draw_hud(img, latency_ms, fps):
    cv2.rectangle(img, (0, 0), (300, 110), (0, 0, 0), -1)
    
    col_lat = (0, 255, 0) if latency_ms < 15 else ((0, 255, 255) if latency_ms < 30 else (0, 0, 255))
    cv2.putText(img, f"LATENCY: {latency_ms:.2f} ms", (10, 30), 0, 0.7, col_lat, 2)
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 60), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "LIVE CAMERA", (10, 90), 0, 0.6, (0, 200, 255), 2)

def set_jetson_performance():
    """Force le mode performance maximale sur la Jetson Orin."""
    subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)
    subprocess.run(['sudo', 'jetson_clocks'], check=True)
        
   



# main program
def run():
    

    gc.disable()
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream()

    
    
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    print("Chargement Moteurs...")
    yolo_model = YOLO(YOLO_ENGINE, task='detect') 
    
    # Warmup
    print("Warmup...")
    dummy_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
    for _ in range(10):
        yolo_model(dummy_img, imgsz=YOLO_IMG_SIZE, verbose=False)
    torch.cuda.synchronize()

    trt_seg = TRTModuleStatic(TWINLITE_ENGINE, device)
    trt_cls = TRTModuleStatic(CLASSIFIER_ENGINE, device)
    cls_resize = T.Resize((32, 32))
    
    # --- UTILISER LA CAMERA USB ---
    reader = USBCameraReader(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    )
    time.sleep(0.5)  # Laisser la caméra s'initialiser

    print(" Ready")

    # Pour calculer le FPS réel
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    while reader.more():
        frame = reader.read()
        if frame is None: 
            continue
        
        display_frame = frame.copy()

        # --- MESURE DE LATENCE ---
        start_evt.record(stream)

        with torch.cuda.stream(stream):
            # 1. Preprocess
            t_frame = torch.from_numpy(frame).to(device, non_blocking=True)
            t_frame_float = t_frame.permute(2, 0, 1).unsqueeze(0).float().div(255.0)

            # 2. YOLO
            results = yolo_model(frame, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

            # 3. TWINLITE
            t_seg_input = torch.nn.functional.interpolate(t_frame_float, size=(360, 640), mode='bilinear')
            seg_outs = trt_seg.forward(t_seg_input, stream)
            da_predict, ll_predict = seg_outs[0], seg_outs[1]

            # 4. PREPA CLASSIFIEUR
            traffic_light_boxes = []
            batch_tensors = []
            
            boxes = results.boxes
            if boxes.shape[0] > 0:
                if (boxes.cls == 9).any():
                    light_boxes = boxes.xyxy[boxes.cls == 9].cpu().numpy()
                    for box in light_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(reader.w, x2), min(reader.h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            crop = t_frame_float[0, :, y1:y2, x1:x2]
                            crop = crop[[2, 1, 0], :, :] 
                            
                            crop = cls_resize(crop)
                            batch_tensors.append(crop)
                            traffic_light_boxes.append((x1, y1, x2, y2))
            
            # 5. CLASSIFIEUR
            run_cls = False
            if batch_tensors:
                run_cls = True
                batch_input = torch.stack(batch_tensors)
                batch_input = (batch_input - 0.5) / 0.5
                cls_out = trt_cls.forward(batch_input, stream)[0]
                scores, idxs = torch.max(torch.softmax(cls_out, dim=1), 1)
                scores = scores.cpu()
                idxs = idxs.cpu()

        end_evt.record(stream)
        stream.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)

        # --- VISUALISATION ---
        mask_da = da_predict[0, 1].cpu().numpy()
        mask_ll = ll_predict[0, 1].cpu().numpy()
        mask_da = cv2.resize(mask_da, (reader.w, reader.h), interpolation=cv2.INTER_NEAREST)
        mask_ll = cv2.resize(mask_ll, (reader.w, reader.h), interpolation=cv2.INTER_NEAREST)
        
        display_frame[mask_da > 0.5] = (display_frame[mask_da > 0.5] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        display_frame[mask_ll > 0.5] = (display_frame[mask_ll > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        cls_results_map = {}
        if run_cls:
            scores_np = scores.numpy()
            idxs_np = idxs.numpy()
            c_map = {1: 'Green', 2: 'Red', 3: 'Yellow'}

            for k in range(len(traffic_light_boxes)):
                name = c_map.get(idxs_np[k])
                s = scores_np[k]
                if name and s > CLS_CONF_THRESH:
                    cls_results_map[k] = (name, s)

        light_counter = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            if cls_id == 9: 
                if light_counter in cls_results_map:
                    n, s = cls_results_map[light_counter]
                    draw_labeled_box(display_frame, x1, y1, x2, y2, f"{n} {s:.2f}", COLORS.get(n, (255,255,255)))
                light_counter += 1
            elif cls_id == 2: 
                draw_labeled_box(display_frame, x1, y1, x2, y2, f"Car {conf:.2f}", COLORS['Car'])
            elif cls_id == 11: 
                draw_labeled_box(display_frame, x1, y1, x2, y2, f"STOP {conf:.2f}", COLORS['Stop'])
            elif cls_id == 0: 
                draw_labeled_box(display_frame, x1, y1, x2, y2, f"Person {conf:.2f}", COLORS['Person'])

        # Calcul du FPS réel
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        draw_hud(display_frame, latency_ms, current_fps)
        # Affichage
        cv2.imshow("Live Camera - Press 'q' to quit", display_frame)
        if cv2.waitKey(1) == ord('q'): 
            break

    reader.stop()
    cv2.destroyAllWindows()
    gc.enable()

if __name__ == "__main__":
    #set_jetson_performance()
    run()