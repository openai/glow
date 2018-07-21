from model import encode, manipulate_range, mix_range
from align_face import align
import numpy as np
from imageio import  mimwrite, get_writer
from PIL import Image

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

# Reshape multiple images to a grid
# def reshape(img,h,w,mirror=False):
#     img = np.reshape(img, [h, w, 256, 256, 3])
#     img = np.transpose(img, [0,2,1,3,4])
#     if mirror:
#         img = img[:,:,::-1,:,:] ## reflect width wise
#     img = np.reshape(img, [h*256, w*256, 3])
#     return img

def resize(arr, res, ratio=1.):
    shape = (int(res*ratio),res)
    return np.array(Image.fromarray(arr).resize(shape, resample=Image.ANTIALIAS))

def make_loop(imgs, gap=10):
    return [imgs[0]]*gap + imgs + [imgs[-1]]*2*gap + imgs[::-1] + [imgs[0]]*gap

def write(imgs, name, fps):
    writer = get_writer(name, fps=fps, quality=6)
    for t in range(len(imgs)):
        writer.append_data(imgs[t])
    writer.close()

def make_video(name, imgs, fps=30, res=1024):
    imgs = [resize(img, res) for img in imgs]
    write(imgs, name + '.mp4', fps)
    imgs = make_loop(imgs)
    write(imgs, name + '_loop.mp4', fps)
    return

def get_manipulations(name, typ, points=46, scale=1.0):
    img = align(name)
    z = encode(img)
    imgs, _ = manipulate_range(z, typ, points, scale)
    return imgs

def get_mixs(name1, name2, points=46):
    img1 = align(name1)
    img2 = align(name2)
    z1 = encode(img1)
    z2 = encode(img2)
    imgs, _ = mix_range(z1, z2, points)
    return imgs

if __name__ == '__main__':
    n1 = 'web/media/geoff.png'
    n2 = 'web/media/leo.png'
    tag = 'Smiling'
    print('Making smiling video')
    imgs_manipulated = get_manipulations(n1, _TAGS.index(tag))
    print('Saving smiling video')
    make_video('geoff_%s' % tag, imgs_manipulated)
    print('Making mixing video')
    imgs_mixed = get_mixs(n1, n2)
    print('Saving mixing video')
    make_video('geoff_leo', imgs_mixed)