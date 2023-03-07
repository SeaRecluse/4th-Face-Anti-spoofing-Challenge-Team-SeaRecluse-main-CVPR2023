import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './checkpoint.pth.tar'

class_nums = 2
dev_path= "./orig_data/"
dev_txt = "dev.txt"
test_txt = "test.txt"
INPUT_SIZE = 224
INPUT_CHNS = 3
#===================================================================
model = timm.create_model('convnext_base_in22ft1k',pretrained=False, num_classes=2)
checkpoint = torch.load(model_path, map_location=device)

state_dict = model.state_dict()
for (k,v) in checkpoint['state_dict'].items():
    if "module." in k:
        key = k[7:]
        print(key)
        state_dict.update({key : v})
    else:
        state_dict.update({k : v})

model.load_state_dict(state_dict)
model.to(device = device)
model.eval()
#===================================================================
trans_test = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.2254, 0.225])
            ])
softmax = nn.Softmax(dim=1)

def as_num(x):
     y='{:.6f}'.format(x)
     return y

def getPred(img):
    input_img = Image.fromarray(img)
    input_tensor = trans_test(input_img).to(device = device)
    input_tensor = input_tensor.view(1, INPUT_CHNS, INPUT_SIZE, INPUT_SIZE)
    return softmax(model(input_tensor)).detach().cpu().numpy()[0]
   
def clearRes(save_path):
    if os.path.exists(save_path):
        os.remove(save_path)

def writeRes(load_path, save_path, use_tta=True):
    line_list = []
    with open(load_path, "r") as f:
        line_list = f.readlines()
    
    save_txt = open(save_path, "a", newline = "\n")
    for n in tqdm(range(len(line_list))):
        per_line = line_list[n].replace("\n", "")
        if per_line:
            img = cv2.imread(dev_path+per_line)
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            output = getPred(img)
            if use_tta:
                img_flip = cv2.flip(img, 1)
                output_flip = getPred(img_flip)
                out = as_num((output[1] + output_flip[1]) / 2)
            else:
                out = as_num(output[1])
            save_txt.write(per_line + " " + out + "\n")
    save_txt.close()

if __name__ == '__main__':
    save_txt_path = "./test-submit.txt"
    clearRes(save_txt_path)
    writeRes(dev_path + dev_txt, save_txt_path)
    writeRes(dev_path + test_txt, save_txt_path)

