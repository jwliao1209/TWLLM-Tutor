#!/bin/bash

if [ ! -d model_weight ]; then
    gdown https://drive.google.com/uc?id=1u8LKk4mTUTCRp1oZx-8vVxwEaewYt -O model_weight.zip
fi

if [ ! -d model_weight ]; then
    unzip model_weight.zip
fi
