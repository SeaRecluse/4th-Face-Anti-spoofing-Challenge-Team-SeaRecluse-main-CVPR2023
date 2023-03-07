import os
import shutil
from tqdm import tqdm

to_path = "./data_div/"
train_list_path ="./train_label.txt"
dev_list_path ="./dev_label.txt"

if os.path.exists(to_path):
	shutil.rmtree(to_path)
os.makedirs(to_path)
os.makedirs(to_path + "train/0/")
os.makedirs(to_path + "train/1/")
os.makedirs(to_path + "val/0/")
os.makedirs(to_path + "val/1/")

def split_train(train_list_path):
	with open(train_list_path, "r") as f:
		img_label_list = f.readlines()
	
	print("Train is moving!")
	for i in tqdm(range(len(img_label_list))):
		img_path = img_label_list[i].split(" ")[0]
		if "0" in img_label_list[i].split(" ")[1]:
			shutil.copyfile(img_path, to_path + "train/0/train+" + img_path.split("/")[1]+"+"+ img_path.split("/")[2]+"+"+ img_path.split("/")[3])
		if "1" in img_label_list[i].split(" ")[1]:
			shutil.copyfile(img_path, to_path + "train/1/train+" + img_path.split("/")[1]+"+"+ img_path.split("/")[2]+"+"+ img_path.split("/")[3])

def split_dev(dev_list_path):
	with open(dev_list_path, "r") as f:
		img_label_list = f.readlines()
	real_list = []
	fake_list = []
	for i in range(len(img_label_list)):
		img_path = img_label_list[i].split(" ")[0]
		if "0" in img_label_list[i].split(" ")[1]:
			fake_list.append(img_path)
		if "1" in img_label_list[i].split(" ")[1]:
			real_list.append(img_path)

	print("Dev-Fake is moving!")
	for i in tqdm(range(len(fake_list))):
		if i <len(fake_list) *2/3:
			shutil.copyfile(fake_list[i],to_path + "train/0/" + fake_list[i].split("/")[0]+"+"+ fake_list[i].split("/")[1]+"+"+ fake_list[i].split("/")[2])
		else:
			shutil.copyfile(fake_list[i],to_path + "val/0/" + fake_list[i].split("/")[0]+"+"+ fake_list[i].split("/")[1]+"+"+ fake_list[i].split("/")[2])
		
	print("Dev-Real is moving!")	
	for i in tqdm(range(len(real_list))):
		if i <len(real_list) *2/3:
			shutil.copyfile(real_list[i],to_path + "train/1/" + real_list[i].split("/")[0]+"+"+ real_list[i].split("/")[1]+"+"+ real_list[i].split("/")[2])
		else:
			shutil.copyfile(real_list[i],to_path + "val/1/" +real_list[i].split("/")[0]+"+"+ real_list[i].split("/")[1]+"+"+ real_list[i].split("/")[2])	

if __name__ == "__main__":
	split_train(train_list_path)
	split_dev(dev_list_path)
	print("Data processing is complete!")