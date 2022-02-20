# environment setting
1. install python 3.9 with https://www.python.org/downloads/release/python-3910/
2. add python to environmetn variable

3. install cuda  with https://developer.nvidia.com/cuda-downloads
4. "nvcc -V" command to check cuda version

5. install pytorch with command "pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html" from web sit "https://pytorch.org/get-started/locally/"

6. check if everything is ok
import torch
print(torch.rand(5, 3))
print(torch.cuda.is_available())


## some import package reference
# argparse
parse_args() function in train_search5cell.py
https://dboyliao.medium.com/python-%E8%B6%85%E5%A5%BD%E7%94%A8%E6%A8%99%E6%BA%96%E5%87%BD%E5%BC%8F%E5%BA%AB-argparse-4eab2e9dcc69
# os
os.path.exist()
os.makedir()
https://www.runoob.com/python/os-mkdir.html