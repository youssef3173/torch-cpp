# Transfer Pytorch Model from Python to C++:
Training a AlexNet in python and use the parameters of this model in C++
# Requirements:
    - python 3.7+
    - pytorch
    - matplotlib
    - libtorch
    - opencv

# Inference Phase:
## build the application:
```
mkdir -p build && cd build
```
```
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```
```
make 
```

## run the application:
```
./torch_inf ../traced_resnet_model.pt ./4.jpg
```
