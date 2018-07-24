# To get x.npy, attr.npy, and z.npy, run in command line
# curl https://storage.googleapis.com/glow-demo/celeba-hq/x.npy > x.npy
# curl https://storage.googleapis.com/glow-demo/celeba-hq/attr.npy > attr.npy
# curl https://storage.googleapis.com/glow-demo/celeba-hq/z.npy > z.npy

import pickle
import numpy as np
import model
from align_face import align_face
from PIL import Image
from tqdm import tqdm

# Align input images
def get_aligned(img_paths):
    xs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.convert('RGB')  # if image is RGBA or Grayscale etc
        img = np.array(img)
        x, face_found = align_face(img)
        if face_found:
            xs.append(x)
    x = np.concatenate(xs, axis=0)
    return x

# Input data. 30000 aligned images of shape 256x256x3
# x = get_aligned(img_paths)
x = np.load('x.npy')
print("Loaded inputs")

# Encode all inputs
def get_z(x):
    bs = 10
    x = x.reshape((-1, bs, 256, 256, 3))
    z = []
    for _x in tqdm(x):
        z.append(model.encode(_x))
    z = np.concatenate(z, axis=0)
    return z

# z = get_z(x)
z = np.load('z.npy')
print("Got encodings")

# Get manipulation vector based on attribute
attr = np.load('attr.npy')

def get_manipulator(index):
    z_pos = [z[i] for i in range(len(x)) if attr[i][index] == 1]
    z_neg = [z[i] for i in range(len(x)) if attr[i][index] == -1]

    z_pos = np.mean(z_pos, axis=0)
    z_neg = np.mean(z_neg, axis=0)
    return z_pos - z_neg

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

z_manipulate = [get_manipulator(i) for i in range(len(_TAGS))]
z_manipulate = 1.6 * np.array(z_manipulate, dtype=np.float32)
print("Got manipulators")
np.save('z_manipulate.npy', z_manipulate)
