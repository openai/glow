apt update && \
apt install -y locales cmake libsm6 libxext6 libxrender-dev && \
locale-gen en_US.UTF-8

export LC_ALL=en_US.UTF-8

# Pip packages for running server and face-aligned (dlib takes a while to install)
pip install flask flask_cors tqdm opencv-python imutils dlib imageio

# Get model weights
curl https://storage.googleapis.com/glow-demo/large3/graph_optimized.pb > graph_optimized.pb

# Get manipulation vectors
curl https://storage.googleapis.com/glow-demo/z_manipulate.npy > z_manipulate.npy

# Get facial landmarks for aligning input faces
curl https://storage.googleapis.com/glow-demo/shape_predictor_68_face_landmarks.dat > shape_predictor_68_face_landmarks.dat

# Pip package for running optimized model with fused kernels
curl https://storage.googleapis.com/glow-demo/blocksparse-1.0.0-py2.py3-none-any.whl > blocksparse-1.0.0-py2.py3-none-any.whl
pip install blocksparse-1.0.0-py2.py3-none-any.whl

# If blocksparse doesn't install, use unoptimized model (and set optimized=False in model.py)
# curl https://storage.googleapis.com/glow-demo/large3/graph_unoptimized.pb > graph_unoptimized.pb

