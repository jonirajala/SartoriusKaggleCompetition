import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from PIL import Image

HEIGHT = 520
WIDTH = 704
IMAGE_PATH = "train/"



# def get_box(a_mask):
#     ''' Get the bounding box of a given mask '''
#     pos = np.where(a_mask)
#     xmin = np.min(pos[1])
#     xmax = np.max(pos[1])
#     ymin = np.min(pos[0])
#     ymax = np.max(pos[0])
#     return [xmin, ymin, xmax, ymax]

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)

df = pd.read_csv('train.csv')
annotations_df = pd.read_csv('train.csv')['annotation']
images_df = pd.read_csv('train.csv')['id']

gb = df.groupby('id')

for i in gb.groups.items():
	print(i)

image_added = False
masks = np.zeros((HEIGHT, WIDTH))
annotations = []

for encoded_mask, image_path in zip(annotations_df, images_df):
	
	if not image_added:
		image = np.array(Image.open(IMAGE_PATH + str(image_path) + ".png"), dtype= np.uint8)
		image_added = True
		plt.imshow(image, cmap='bone')
	if image_path == images_df.iloc[0]:
		decoded_mask = rle_decode(encoded_mask, (HEIGHT, WIDTH))
		annotations.append((decoded_mask))
		masks = np.logical_or(masks, decoded_mask)

		# image = Image.fromarray(decoded_mask)
		# box = get_box(decoded_mask)
		# box_width = box[2]-box[0]
		# box_heigth = box[1]-box[3]
		# rect = patches.Rectangle((box[0], box[3]), box_width, box_heigth, linewidth=1, edgecolor='r', facecolor='none')
		# plt.imshow(decoded_mask, alpha=0.3)
	else:
		break
# plt.imshow(masks, alpha=0.3)
# plt.show()

