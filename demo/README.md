Code for the demo used in blog post.

# Setup
Run `./script.sh`.

The script install pip packages, downloads pretrained model weights,
manipulation vectors and a facial landmarks detector for aligning input faces.

# Using pre-trained model
To use pre-trained CelebA-HQ model for encoding/decoding/manipulating images, use `model.py`

If your image is not aligned, use `align_face.py` to align image.

To create videos, check `videos.py`

# Create manipulation vectors for an attribute of your choice
Scrape images from the internet for an attribute of your choice (say red-hair vs not red-hair). Then, to obtain manipulation vectors from, use `get_manipulators.py`

To see how it was done for the CelebA-HQ dataset (which has 40 attributes),
first download the input images (x.npy), their attributes (attr.npy) and their encoding (z.npy)
```
curl https://storage.googleapis.com/glow-demo/celeba-hq/x.npy > x.npy
curl https://storage.googleapis.com/glow-demo/celeba-hq/attr.npy > attr.npy
curl https://storage.googleapis.com/glow-demo/celeba-hq/z.npy > z.npy
```
Then, run `get_manipulators.py`

# Run server and client for demo
To run server for demo, run `python server.py`.

To run client, run `python -m http.server` (starts a local http server at port 8000) and open `0.0.0.0:8000/web` in your browser.

To test client, upload `test/img.png`. You should see aligned image and be able to move sliders.

