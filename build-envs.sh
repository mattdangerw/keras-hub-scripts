conda env remove -n keras-nlp-cpu || true
conda create --name keras-nlp-cpu python=3.11 -y
conda activate keras-nlp-cpu
pip install -r requirements.txt
pip install --no-deps -e "."
export KERAS_BACKEND=jax

conda env remove -n keras-nlp-torch || true
conda create --name keras-nlp-torch python=3.11 -y
conda activate keras-nlp-torch
pip install -r requirements-torch-cuda.txt
pip install --no-deps -e "."
export KERAS_BACKEND=torch

conda env remove -n keras-nlp-jax || true
conda create --name keras-nlp-jax python=3.11 -y
conda activate keras-nlp-jax
pip install -r requirements-jax-cuda.txt
pip install --no-deps -e "."
export KERAS_BACKEND=jax

conda env remove -n keras-nlp-tensorflow || true
conda create --name keras-nlp-tensorflow python=3.11 -y
conda activate keras-nlp-tensorflow
pip install -r requirements-tensorflow-cuda.txt
pip install --no-deps -e "."
export KERAS_BACKEND=tensorflow

conda env remove -n keras-nlp-2.15 || true
conda create --name keras-nlp-2.15 python=3.11 -y
conda activate keras-nlp-2.15
pip install -r requirements-common.txt
pip install tensorflow-text==2.15 tensorflow==2.15 keras-core
pip install --no-deps -e "."
export KERAS_BACKEND=tensorflow
