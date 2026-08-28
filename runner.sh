#!/bin/bash
unzip -o /content/geodiff3d_payload.zip -d /content
pip install -r /content/inference/requirements_gpu.txt
python /content/inference/gpu_pipeline.py
zip -r /content/output.zip /content/inference/output
