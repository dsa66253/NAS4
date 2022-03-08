# pytorch and cuda setting
1. install python 3.9 with https://www.python.org/downloads/release/python-3910/
2. add python to environmetn variable

3. install cuda  with https://developer.nvidia.com/cuda-downloads
4. "nvcc -V" command to check cuda version

5. install pytorch with command "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html" from web sit "https://pytorch.org/get-started/locally/"

6. check if everything is ok
import torch
print(torch.rand(5, 3))
print(torch.cuda.is_available())

# related package installation
1. make sure requirements.txt being in root directory
2. running command "pip install -r requirements.txt" to install related package


# some import package reference
## argparse
parse_args() function in ./train_search5cell.py
https://dboyliao.medium.com/python-%E8%B6%85%E5%A5%BD%E7%94%A8%E6%A8%99%E6%BA%96%E5%87%BD%E5%BC%8F%E5%BA%AB-argparse-4eab2e9dcc69
## os
os.path.exist() function in ./feature/moake_dir.py
os.makedir()
https://www.runoob.com/python/os-mkdir.html

## pathlib
path in train_search5cell.py
https://myapollo.com.tw/zh-tw/python-pathlib/

## radom.seed()
https://www.geeksforgeeks.org/random-seed-in-python/

## torch.cuda.manual_seed()
https://blog.csdn.net/sdnuwjw/article/details/106467809

## cudnn.benchmark in ./train_search_5cell.py
https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936



## model(input)
calling instance behave like calling function
https://www.geeksforgeeks.org/__call__-in-python/

## lambda function in ./models/alpha/operation.py
https://www.learncodewithmike.com/2019/12/python-lambda-functions.html

## nn.BatchNorm2d() in ./models/alpha/operation.pn
過activation function前，先過個加速收斂速度
https://zhuanlan.zhihu.com/p/69431151


## torch.nn.init.kaiming_normal_ in ./models/alpha/operation.pn
初始化某layer的function
https://zhuanlan.zhihu.com/p/53712833


## tensorboard in ./train_search_5cell.py
use this to examine loss and model afterward
https://pytorch.org/docs/stable/tensorboard.html
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

## tqdm in ./train_search_5cell.py
https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/


## hasattr() ./models/nas_5cell.py
to check if object has certain atttribute
https://www.runoob.com/python/python-func-hasattr.html

## named_parameters() ./models/nas_5cell.py
it return the name of layer and parameters
and make it a list to print. eg print(list(model.named_parameters()))
eg for name, para in myModel.named_parameters():
https://blog.csdn.net/qq_36530992/article/details/102729585
https://pytorch.org/docs/stable/generated/torch.nn.Module.html

## Autograd tutorial
understand required grad
3 ways to no track gradient
1. x.requires_grad_(False)
2. x.detach()
3. with torch.nod_grad():
https://www.youtube.com/watch?v=DbeIqrwb_dE&ab_channel=PythonEngineer

## register_parameter() ./models/nas_5cell.py
add parameter to a model
https://stackoverflow.com/questions/59234238/how-to-add-parameters-in-module-class-in-pytorch-custom-model

## zip() in ./models/nas_5cell.py
https://ithelp.ithome.com.tw/articles/10218029


## add_module("name", nn.module) in ./models/nas_5cell.py
這兩個搭配起來可以讓layer可以自定義名稱，之後update參數時會用這個來決定是alpha or weight
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
