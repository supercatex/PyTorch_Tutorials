import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import cv2

os.environ['TORCH_HOME'] = '.'
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(dev)
with open("coco_classes.txt", "r") as f:
    classes = [x.strip() for x in f.readlines()]

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.as_tensor(image, dtype=torch.float32)
    image = image / 255
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)
    image = image.to(dev)
    out = model(image)
    o_boxes = out[0]["boxes"]
    o_labels = out[0]["labels"]
    o_scores = out[0]["scores"]
    for box, label, score in zip(o_boxes, o_labels, o_scores):
        if score < 0.7: continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        print(label, classes[label - 1])
    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()