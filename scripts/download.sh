#!/bin/bash

if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=1UC0wrxeLv9Lf5rJS2S2GvDVzakP592Yl -O data.zip
fi

if [ ! -d data ]; then
    unzip data.zip
fi

if [ ! -d model_weight ]; then
    gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O model_weight.zip
fi

if [ ! -d model_weight ]; then
    unzip model_weight.zip
fi
