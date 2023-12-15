#!/bin/bash

if [ ! -d model_weight ]; then
    gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O model_weight.zip
fi

if [ ! -d model_weight ]; then
    unzip model_weight.zip
fi
