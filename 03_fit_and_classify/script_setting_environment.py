set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        scikit-image \
        scikit-learn \
        matplotlib \
        h5py \
        keras \
        tensorflow
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'

    mkdir ~/.keras
    echo '{"backend": "tensorflow",\n "epsilon": 1e-07,\n "floatx": "float32"}' > ~/.keras/keras.json
    python3 -c 'from keras.applications import ResNet50, InceptionV3, Xception;\
                model = ResNet50(weights="imagenet", include_top=False);\
                model = InceptionV3(weights="imagenet", include_top=False);\
                model = Xception(weights="imagenet", include_top=False);'
}

"$@"