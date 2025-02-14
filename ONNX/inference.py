import onnx 
import onnxruntime as ort
import onnx_graphsurgeon as gs
import numpy as np
from scipy.special import softmax
import torch
import torchvision.transforms as transforms
import time
from PIL import Image
from pathlib import Path
import argparse

def create_session(model_path):
    
    onnx_sesion = ort.InferenceSession(model_path)
    graph= gs.import_onnx(onnx.load(model_path))
    input_shape = graph.inputs[0].shape
    return onnx_sesion,input_shape

def process_input(input_data,input_shape):
    
    h,w=input_shape[2],input_shape[3]

    if input_data.suffix =='.npy':
        input_data = np.load(input_data)
        
    elif input_data.suffix .lower() =='.jpg':
        
        input_data = Image.open(input_data)
        preprocess = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
        ])
        input_data = preprocess(input_data)
        input_data = np.expand_dims(input_data,axis=0)
        
    else:
        raise ValueError("Input data format not supported")
        
    return input_data


def run_inference(onnx_session,input_data):
    
    start=time.time()
    inputs={"input":input_data}
    output=onnx_session.run(None,inputs)
    end=time.time()
    
    print(f"Output: {np.argmax(softmax(output))}")
    print(f"Inference time: {end-start} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on ONNX model')
    parser.add_argument('-m','--model_path', type=str, help='Path to ONNX model')
    parser.add_argument('-i','--input_data', type=str, help='Path to input data')
    return parser.parse_args()


if __name__ == '__main__':
    args=parse_args()
    
    MODEL_PATH = Path(args.model_path)
    INPUT_DATA = Path(args.input_data)
    
    
    print( f"Model Path: {MODEL_PATH}")
    print( f"Input Data Path: {INPUT_DATA}")
    
    print("Running inference..")
    
    onnx_session,input_shape = create_session(MODEL_PATH)
    
    
    input_data = process_input(INPUT_DATA,input_shape)
    
    run_inference(onnx_session,input_data)
    
    
    print("Inference completed")
    
    
