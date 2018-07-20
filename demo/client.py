import requests
import asyncio
import concurrent.futures

from PIL import Image
import numpy as np
import base64
import time

host = '0.0.0.0:5050'

_TAGS = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
_TAGS = _TAGS.split()

aligned_dir = 'static/raw_aligned/'
results_dir = 'static/results/'
raw_dir = 'static/raw/'


def deserialise_img(img_str):
    img = base64.b64decode(img_str.split(",")[-1])
    img = Image.open(BytesIO(img))
    img = img.convert('RGB')
    img = np.array(img)
    return img


def serialise_img(arr):
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')


def deserialise_nparr(arr_str):
    arr = np.loads(base64.b64decode(arr_str))
    return np.array(arr, dtype=np.float32)


def serialise_nparr(arr):
    arr = np.array(arr, dtype=np.float16)
    return base64.b64encode(arr.dumps()).decode('utf-8')


def encode_file(fname):
    img = Image.open(fname)
    img = np.reshape(np.array(img), [1, 256, 256, 3])
    z = encode(img)
    return z


def align_encode(img):
    img = serialise_img(img)
    t = time.time()
    response = requests.post(host + '/api/align_encode', json={'img': img})
    print("Response", time.time() - t)
    img, z = response.json().get('img'), response.json().get('z')
    return deserialise_img(img), deserialise_nparr(z)


def encode(img):
    img = serialise_img(img)
    t = time.time()
    response = requests.post(host + '/api/encode', json={'img': img})
    print("Response", time.time() - t)
    t = time.time()
    z = response.json().get('z')
    print("Get from json time", time.time() - t)
    return deserialise_nparr(z)


def decode(z):
    z = serialise_nparr(z)
    t = time.time()
    response = requests.post(host + '/api/decode', json={'z': z})
    print("Response", time.time() - t)
    t = time.time()
    img = response.json().get('img')
    print("Get from json time", time.time() - t)
    return deserialise_img(img)


def manipulate_range(z, typ, points):
    z = serialise_nparr(z)
    response = requests.post(
        host + '/api/manipulate_range', json={'z': z, 'typ': typ, 'points': points})
    imgs = response.json().get('img')
    return deserialise_nparr(imgs)


def mix_range(z1, z2, points):
    z1 = serialise_nparr(z1)
    z2 = serialise_nparr(z2)
    response = requests.post(host + '/api/mix_range',
                             json={'z1': z1, 'z2': z2, 'points': points})
    imgs = response.json().get('img')
    return deserialise_nparr(imgs)


def test_enc_dec():
    z = encode_file(aligned_dir + '1743.png')
    img = decode(z)
    img = Image.fromarray(img[0])
    img.save(aligned_dir + '1743_dec.png')


def test_manipulate():
    z = encode_file(aligned_dir + '1743.png')
    typ = _TAGS.index('Smiling')
    imgs = manipulate_range(z, typ, 5)
    imgs = Image.fromarray(np.concatenate(
        [imgs[i] for i in range(len(imgs))], axis=1))
    imgs.save(results_dir + 'manipulate_%s.png' % str(typ))


def test_interpolate():
    z1 = encode_file(aligned_dir + '1743.png')
    z2 = encode_file(aligned_dir + '1778.png')
    imgs = mix_range(z1, z2, 5)
    imgs = Image.fromarray(np.concatenate(
        [imgs[i] for i in range(len(imgs))], axis=1))
    imgs.save(results_dir + 'interpolate.png')


def test_align_encode():
    img = Image.open(raw_dir + 'beyonce.jpg')
    img = np.array(img)
    img, z = align_encode(img)
    img = Image.fromarray(img)
    img.save(aligned_dir + 'beyonce_align.png')


def test_speed(batch_size=1):
    img = Image.open(aligned_dir + '1743.png')
    img = np.tile(np.reshape(np.array(img), [1, 256, 256, 3]), [
                  batch_size, 1, 1, 1])

    t = time.time()
    z = encode(img)
    print("Time to encode first response", time.time() - t, z.shape)
    t = time.time()
    for i in range(10):
        z = encode(img)
    print("Encoding latency {} sec/img".format((time.time() - t) / (batch_size * 10)))

    t = time.time()
    img = decode(z)
    print("Time to decode first response", time.time() - t, img.shape)
    t = time.time()
    for i in range(10):
        img = decode(z)
    print("Decoding latency {} sec/img".format((time.time() - t) / (batch_size * 10)))


def test_batch(batch_size=10):
    img = Image.open(aligned_dir + '1743.png')
    img = np.tile(np.reshape(np.array(img), [1, 256, 256, 3]), [
                  batch_size, 1, 1, 1])

    t = time.time()
    z = encode(img)
    print("Time to encode first response", time.time() - t, z.shape)
    t = time.time()
    for i in range(1):
        z = encode(img)
    print("Encoding latency {} sec/img".format((time.time() - t) / (batch_size * 1)))


def test_async(batch_size=1):

    img = Image.open(aligned_dir + '1743.png')
    img = np.tile(np.reshape(np.array(img), [1, 256, 256, 3]), [
                  batch_size, 1, 1, 1])
    t = time.time()
    z = encode(img)
    print("Time to encode first response", time.time() - t, z.shape)

    def _encode(i):
        time.sleep(i * 0.5)
        encode(img)

    async def main():
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(
                executor, _encode, i) for i in range(20)]
            for response in await asyncio.gather(*futures):
                pass

    loop = asyncio.get_event_loop()
    t = time.time()
    loop.run_until_complete(main())
    print("Encoding latency {} sec/img".format((time.time() - t) / (batch_size * 20)))
