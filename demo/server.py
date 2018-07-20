import model
from align_face import align_face
from flask import Flask, jsonify, request
from flask_cors import CORS

import base64
import time
import numpy as np
from PIL import Image
from io import BytesIO
app = Flask(__name__)
CORS(app)


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


def send(result):
    # img, z are batches, send as list of singles
    img, z = result
    # , z=list(map(serialise_nparr, z)))
    return jsonify(img=list(map(serialise_img, img)))


def send_proj(result, proj):
    # img, z are batches, send as list of singles
    img, z = result
    return jsonify(face_found=True, img=list(map(serialise_img, img)), z=list(map(serialise_nparr, z)), proj=proj.tolist())


def get(request, key):
    return request.get_json().get(key)


def get_z(request, key):
    # z is a single point, batch it for use
    z = get(request, key)
    return np.expand_dims(deserialise_nparr(z), axis=0)


@app.route('/')
def hello_world():
    return 'Welcome to Glow!'

# Align and encode image
#
# args
# img: Image as base64 string
#
# returns
# json: {'face_found': face_found, 'img':[base64 img], 'z': [serialised z]}
@app.route('/api/align_encode', methods=['POST'])
def align_encode():
    r = request
    img = get(r, 'img')
    # img = parse_img(img) if in jpg etc format
    img = deserialise_img(img)
    img, face_found = align_face(img)
    if face_found:
        img = np.reshape(img, [1, 256, 256, 3])
        print(img.shape)
        z = model.encode(img)
        proj = model.project(z)  # get projections. Not used
        result = img, z
        # jsonify(img=serialise_img(img), z=serialise_nparr(z))
        return send_proj(result, proj)
    else:
        return jsonify(face_found=False)

# Maipulate single attribute
#
# args
# z: Serialised np array for encoding of image
# typ: int in [0,40), representing which attribute to manipulate
# alpha: float, usually in [-1,1], representing how much to manipulate. 0 gives original image
#
# returns
# json: {'img': [img]}
@app.route('/api/manipulate', methods=['POST'])
def manipulate():
    r = request
    z = get_z(r, 'z')
    typ = get(r, 'typ')
    alpha = get(r, 'alpha')
    result = model.manipulate(z, typ, alpha)
    return send(result)

# Manipulate multiple attributes
# typs: list of typ
# alphas: list of corresponding alphas
@app.route('/api/manipulate_all', methods=['POST'])
def manipulate_all():
    r = request
    z = get_z(r, 'z')
    typs = get(r, 'typs')
    alphas = get(r, 'alphas')
    result = model.manipulate_all(z, typs, alphas)
    return send(result)

# Mix two faces
#
# args
# z1: Serialised np array for encoding of image 1
# z2: Serialised np array for encoding of image 2
# alpha: float in [0,1], representing how much to mix. 0.5 gives middle image
#
# returns
# json: {'img': [img]}
@app.route('/api/mix', methods=['POST'])
def mix():
    r = request
    z1 = get_z(r, 'z1')
    z2 = get_z(r, 'z2')
    alpha = get(r, 'alpha')
    result = model.mix(z1, z2, alpha)
    return send(result)

# Get random image
@app.route('/api/random', methods=['POST'])
def random():
    r = request
    bs = get(r, 'bs')
    result = model.random(bs)
    img, z = result
    proj = model.project(z)
    return send_proj(result, proj)

# Extra functions
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    z = get_z(r, 'z')
    typs = get(r, 'typs')
    alphas = get(r, 'alphas')  # value between [-1,1] -> 0.5 is original image
    return jsonify(z="")


@app.route('/api/manipulate_range', methods=['POST'])
def manipulate_range():
    r = request
    z = get_z(r, 'z')
    typ = get(r, 'typ')
    points = get(r, 'points')
    result = model.manipulate_range(z, typ, points)
    return send(result)


@app.route('/api/mix_range', methods=['POST'])
def mix_range():
    r = request
    z1 = get_z(r, 'z1')
    z2 = get_z(r, 'z2')
    points = get(r, 'points')
    result = model.mix_range(z1, z2, points)
    return send(result)

# Legacy
# @app.route('/api/encode', methods=['POST'])
# def encode():
#     t = time.time()
#     r = request
#     img = get(r, 'img')
#     print("Time to read from request", time.time() - t)
#     t = time.time()
#     img = deserialise_nparr(img)
#     print("Time to serialise from request", time.time() - t)
#     t = time.time()
#     z = model.encode(img)
#     print("TIme to encode", time.time() - t)
#     t = time.time()
#     json = jsonify(z=serialise_nparr(z))
#     print("Time to jsonify", time.time() - t)
#     return json
#
# @app.route('/api/decode', methods=['POST'])
# def decode():
#     t = time.time()
#     r = request
#     z = get(r, 'z')
#     print("Time to read from request", time.time() - t)
#     t = time.time()
#     z = deserialise_nparr(z)
#     print("Time to serialise from request", time.time() - t)
#     t = time.time()
#     img = model.decode(z)
#     print("TIme to decode", time.time() - t)
#     t = time.time()
#     json = jsonify(img=serialise_img(img))
#     print("Time to jsonify", time.time() - t)
#     return json
#
# @app.route('/api/align', methods=['POST'])
# def align():
#     r = request
#     img = get(r, 'img')
#     # img = parse_img(img) if in jpg etc format
#     img = deserialise_img(img)
#     img = align_face(img)
#     return jsonify(img=serialise_img(img))


# FaceOff! Use for manipulation and blending faces
if __name__ == '__main__':
    print('Running Flask app...')
    app.run(host='0.0.0.0', port=5050)
