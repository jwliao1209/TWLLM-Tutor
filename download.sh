#!/bin/bash

if [ ! -d pretrain ]; then
    gdown https://drive.google.com/uc?id=1qlgYaHXzCLmt_EpdIIIc0i4qu83rfqhp -O pretrain.zip
fi

if [ ! -d pretrain ]; then
    unzip pretrain.zip
fi
