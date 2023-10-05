for backend in "torch" "jax" "tensorflow"; do
    conda env remove -n keras-${backend} || true
    conda create --name keras-${backend} python=3.10 -y
    conda activate keras-${backend}
    pip install -r requirements-${backend}-cuda.txt
    pip install --no-deps -e "."
done
