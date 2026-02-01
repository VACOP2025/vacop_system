import torch
import torch.nn as nn
from ultralytics import YOLO
from TwinLite import TwinLiteNet

# --- CONFIG ---
DEVICE = torch.device('cuda:0')
YOLO_PATH = r'../models/yolo26m.pt'
CLASSIFIER_PATH = "../models/traffic_classifier.pth"
TWINLITE_PATH = "../models/best.pth"

print(" Exporting YOLO to TensorRT (FP16)...")
model = YOLO(YOLO_PATH)
model.export(format='engine', imgsz=320, half=True, device=0, simplify=True, workspace=4)
print("✅ YOLO Exported: yolo26m.engine")

print("\n Exporting TwinLiteNet to ONNX...")
model_seg = TwinLiteNet().to(DEVICE).eval()
state_dict = torch.load(TWINLITE_PATH, map_location=DEVICE)
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model_seg.load_state_dict(new_state_dict)

dummy_input = torch.randn(1, 3, 360, 640).to(DEVICE)
torch.onnx.export(model_seg, dummy_input, "twinlite.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['da', 'll'],
                  dynamic_axes=None)
print(" TwinLiteNet ONNX created. Now creating Engine...")

print("\n Exporting Classifier to ONNX...")
class TrafficLightNet(nn.Module):
    def __init__(self):
        super(TrafficLightNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
    def forward(self, x): return self.classifier(self.features(x))

cls_model = TrafficLightNet().to(DEVICE).eval()
try:
    cls_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
except:
    cls_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False))

dummy_cls = torch.randn(16, 3, 32, 32).to(DEVICE)
torch.onnx.export(cls_model, dummy_cls, "classifier.onnx",
                  opset_version=11,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}) 
print("✅ Classifier ONNX created.")
print("\n⚠️  IMPORTANT : POUR FINIR LA CONVERSION (.onnx -> .engine)")
print("Ouvrez un terminal et lancez ces commandes (nécessite trtexec installé avec TensorRT) :")
print(f'trtexec --onnx=twinlite.onnx --saveEngine=twinlite.engine --fp16')
print(f'trtexec --onnx=classifier.onnx --saveEngine=classifier.engine --fp16 --minShapes=input:1x3x32x32 --optShapes=input:8x3x32x32 --maxShapes=input:32x3x32x32')