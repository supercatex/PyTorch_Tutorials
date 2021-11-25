import torch
from torchvision.models import vgg16
import cv2
import os

os.environ['TORCH_HOME'] = '.'
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)
model.eval()
model.to(dev)
with open("imagenet_classes.txt", "r") as f:
    classes = [x.strip() for x in f.readlines()]

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    h, w, c = frame.shape
    image = frame[:, (w - h) // 2:w - (w - h) // 2, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.as_tensor(image, dtype=torch.float32)
    image = image / 255
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, dim=0)
    image = image.to(dev)
    out = model(image)
    idx = torch.max(out, dim=1).indices
    print(classes[idx])

    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
