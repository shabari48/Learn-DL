# Tasks Done Today (12/2/2025)

- Installed CUDA and setup PyTorch Locally
- Learnt about ONNX , it's usage
- CNN built for leaf dataset ->  ONNX format
- Resnet 50 architecture -> ONNX format
- Learnt about basics of ONNX-GraphSurgeon (Graph,Tensor,Node)
- Tried out inferencing with ONNX model of Resnet 50


# TO DO

 **Have to learn and explore more about ONNX-GraphSurgeon and Inferencing with ONNX Runtime**


# NOTES

ONNX stands for Open Neural Network Exchange

The main benefits ONNX is

- interoperability
- hardware access

Interoperability allows models trained from one framework to converted into ONNX format and we can use that ONNX format to inference or we can convert it back into model in any other framework based on the device where we are going to run the model for Inference




In Pytorch → ONNX

```python
torch.onnx.export( model,dummy_input,onnx_path,input_names["input"],output_names=["output"],
)
```

To Visualize the Model

```bash
netron [modelfile]
```

Installation

```bash
conda install conda-forge::onnx
conda install conda-forge::onnxruntime
pip install onnx-graphsurgeon
```

The whole idea behind this


<https://onnx.ai/about.html>

**open format built to represent machine learning models**

Models are built using libraries → like PyTorch , Tensor flow, Core ml and there will be many libraries like that which we will use to built machine learning models

 So what is ONNX facilitates is show like we will be building a model using this frameworks and then we will convert it into the ONNX format so it acts like an open source format where we can convert the ONNX format into any other framework based on where we are going to deploy it

TensorFlow Lite (TF-Lite) is **an open-source framework for running machine learning models on devices like smartphones, tablets, and IoT devices**. It's designed to perform inference on-device, also known as edge computing.

[**Tensor-RT**](https://developer.nvidia.com/tensorrt?ref=blog.roboflow.com) is a machine learning framework that is published by Nvidia to run inference that is machine learning inference on their hardware. Tensor-RT is highly optimized to run on NVIDIA GPUs and Boards .

TF-Lite ,Tensor-RT → ML frameworks focused on optimizing the model for particular devices

My  Simple understanding

- **PyTorch (or TF, Scikit Learn)** is where the model is built and trained.
- **ONNX** serves as a bridge by providing a standard format to export and transfer the model.
- **Tensor-RT** takes the ONNX model, optimizes it for NVIDIA GPUs, and enables efficient deployment for inference.

Challenges

- Multiple Frameworks for building models based on use cases
- Hardware Accelerators → NVIDIA GPUS, TPUS, INTEL GPUS
- Inferencing → Tensor-RT,TF-Lite, Open-VINO


