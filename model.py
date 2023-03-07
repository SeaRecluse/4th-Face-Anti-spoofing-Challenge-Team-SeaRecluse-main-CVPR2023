import torch
import torchvision
from thop import profile
import timm

# Model
print('==> Building model..')
model=timm.create_model('convnext_xlarge_in22ft1k',pretrained=False, num_classes=2)

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))