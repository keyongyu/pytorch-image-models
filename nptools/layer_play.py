import timm
import torch
from timm.models import load_checkpoint

CHECKPOINT = '/home/keyong/cls2/code/pepsioutput/20260725-113831-tf_efficientnet_lite0_in1k-224/model_best.pth.tar'

#m = timm.create_model('xception41', pretrained=True)
m = timm.create_model('tf_efficientnet_lite0.in1k', num_classes=3, pretrained=False)
load_checkpoint(m, CHECKPOINT, weights_only=False)
m.eval()



from PIL import Image
import torchvision.transforms.functional as F

IMG_PATH = '/home/keyong/cls2/code/pepsi_merge/val/CRM060109/0_4_ - Copy.jpg'
IMG_PATH = '/home/keyong/cls2/code/pepsi_merge/val/CRM060109/0_132_ - Copy.jpg'
IMG_PATH = '/home/keyong/cls2/code/pepsi_merge/val/CRM060118/heineken.jpg'

IMG_PATH = '/home/keyong/cls2/code/posmlv/output/20260728-094659-tf_efficientnet_lite0_in1k-224/checkpoint-41.pth.tar'

# crop_pct=1.0: resize_size = int(224 / 1.0) = 224
# timm resizes the shorter edge to resize_size, then center crops to 224x224
img = Image.open(IMG_PATH).convert('RGB')
img = img.resize((224,224), Image.BICUBIC)  # shorter edge → 224
# center crop to 224x224
w, h = img.size
x = F.to_tensor(img)                          # [3, 224, 224], values 0-1
x = F.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -> [-1, 1]
x = x.unsqueeze(0)                            # [1, 3, 224, 224]
print(f'Input shape: {x.shape}')

with torch.no_grad():
    logits = m(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()

print(f'Logits: {logits}')
print(f'Probs:  {probs}')
print(f'Predicted class index: {pred}')