

from re import M
import torch
import torch.nn as nn
from layer import Layer
from data.config import epoch_to_drop
import torch.nn.functional as F
import os
import numpy as np
from feature.utility import getCurrentTime1 as getCurrentTime
class Model(nn.Module):
    def __init__(self, numOfLayer, numOfInnerCell, numOfClasses):
        super(Model, self).__init__()
        #info private attribute
        self.numOfOpPerCell = 5
        self.numOfLayer = numOfLayer
        self.numOfInnerCell = numOfInnerCell
        self.alphasSaveDir = r'./alpha_pdart_nodrop'
        self.currentEpoch = 0
        self.maskSaveDir = r'./saved_mask_per_epoch'
        #info network structure
        self.layerDict = nn.ModuleDict({
            "layer_0":Layer(self.numOfInnerCell, 0, 3, 96, 4),
            "layer_1":Layer(self.numOfInnerCell, 1, 96, 96, 4),
            "layer_2":Layer(self.numOfInnerCell, 2, 96, 256, 1),
            "layer_3":Layer(self.numOfInnerCell, 3, 256, 256, 1),
            "layer_4":Layer(self.numOfInnerCell, 4, 256, 384, 1),
            "layer_5":Layer(self.numOfInnerCell, 5, 384, 384, 1),
            "layer_6":Layer(self.numOfInnerCell, 6, 384, 384, 1),
            "layer_7":Layer(self.numOfInnerCell, 7, 384, 384, 1),
            "layer_8":Layer(self.numOfInnerCell, 8, 384, 256, 1),
            "layer_9":Layer(self.numOfInnerCell, 9, 256, 256, 1),
            "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        })
        # print(self.layerDict["layer_0"])
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, numOfClasses)
        )
        
        #info initailize alphas and alphas mask, and register them with model
        self.initailizeAlphas()
        self.alphasMask = torch.full(self._alphas.shape, False, dtype=torch.bool) #* True means that element are masked(dropped)
        
        #* self.alphas can get the attribute
    def initailizeAlphas(self):
        #* set the first innercell's alpha to being evenly distributed, set second innercell's alphas to random
        torch.manual_seed(0)
        self._alphas = F.softmax( 0.01*torch.ones([self.numOfLayer, self.numOfInnerCell, self.numOfOpPerCell], requires_grad=False) )
        tmp = F.softmax( torch.rand(self.numOfLayer, self.numOfInnerCell, self.numOfOpPerCell) )
        
        # #* set probability of innerCell with index 1 to random
        # for layer in range(self.numOfLayer):
        #     self._alphas[layer][1] = tmp[layer][1]
        
        self._alphas.requires_grad_(True)
        self.register_parameter("alphas", nn.Parameter(self._alphas))
        
    def dropMinAlpha(self):
        #* drop min alphas at certain epoch
        #* algo: find min for each innerCell and then modify mask
        (_, allMinIndex) = torch.topk(self.alphas, self.numOfOpPerCell, largest=False)
        for layer in range(self.numOfLayer):
            for indexOfInnerCell in range(self.numOfInnerCell):
                minIndexList = allMinIndex[layer][indexOfInnerCell].tolist() #* sorted index from small to large
                # minIndex = minIndexList.pop(0) #* pop the min index from a list
                for minIndex in minIndexList:
                    if self.alphasMask[layer][indexOfInnerCell][minIndex] == False:
                        self.alphasMask[layer][indexOfInnerCell][minIndex] = True
                        break
        print("dropMinAlpha mask", self.alphasMask)
    
    def saveAlphas(self, epoch, kthSeed):
        # print("save alphas")
        if not os.path.exists(self.alphasSaveDir):
            os.makedirs(self.alphasSaveDir)
        tmp = self.alphas.clone().detach()
        tmp = tmp.data.cpu().numpy()
        fileName =  os.path.join(self.alphasSaveDir, 'alpha_prob_' + str(kthSeed) + '_' + str(epoch)+"_"+getCurrentTime())
        np.save(fileName, tmp)
        # print("\nepcho:", epoch, "save alphas:", tmp)
    def saveMask(self, epoch, kthSeed):
        # print("save mask")
        if not os.path.exists(self.maskSaveDir):
            os.makedirs(self.maskSaveDir)
        tmp = self.alphasMask.clone().detach()
        tmp = tmp.data.cpu().numpy()
        fileName =  os.path.join(self.maskSaveDir, 'mask_' + str(kthSeed) + '_' + str(epoch))
        np.save(fileName, tmp)
    
    def normalizeAlphas(self):
        #* set alphas to zero whose mask is true and pass it through softmax
        tmp = self.alphas.clone().detach()
        with torch.no_grad():
            #*inplace operation
            self.alphas *= 0
            self.alphas += F.softmax(tmp, dim=-1)
            # self.alphas= tmp #! this may cause optimizer will update old object
            # self.alphas = torch.nn.Parameter(tmp)
        # print("normalize alphas", self.alphas)
    def filtAlphas(self):
        #* set alphas to zero whose mask is true
        # tmp = self.alphas.clone().detach()
        with torch.no_grad():
            #*inplace operation
            self.alphas[self.alphasMask] = 0
            # tmp = F.softmax(tmp, dim=-1)
            # self.alphas= tmp #! this will cause optimizer to update old object
            # self.alphas = torch.nn.Parameter(tmp)
        # print("normalize alphas", self.alphas)
    def checkDropAlpha(self, epoch, kthSeed):
        #* to check whether drop alphas at particular epoch
        return  epoch in epoch_to_drop
    def forward(self, input, epoch, kthSeed):
        #* every time use alphas need to set alphas to 0 which has been drop
        self.filtAlphas()

        output = self.layerDict["layer_0"](input, self.alphas[0])
        output = self.layerDict["layer_1"](output, self.alphas[1])
        output = self.layerDict["max_pool1"](output)
        output = self.layerDict["layer_2"](output , self.alphas[2])
        output = self.layerDict["layer_3"](output , self.alphas[3])
        output = self.layerDict["max_pool2"](output)
        output = self.layerDict["layer_4"](output , self.alphas[4])
        output = self.layerDict["layer_5"](output , self.alphas[5])
        #! 先關閉layer3 layer4 增加訓練速度
        output = self.layerDict["layer_6"](output , self.alphas[6])
        output = self.layerDict["layer_7"](output , self.alphas[7])
        output = self.layerDict["layer_8"](output , self.alphas[8])
        output = self.layerDict["layer_9"](output , self.alphas[9])
        output = self.layerDict["max_pool3"](output)
        # print("tensor with shape{} is going to fc".format(output.shape))
        output = torch.flatten(output, start_dim=1)
        # print("tensor with shape{} is going to fc".format(output.shape))
        # print("alphas", self.alphas)
        # print("alphas mask", self.alphasMask)
        # print("model", self)
        # exit()
        output = self.fc(output)
        return output
    def getAlphas(self):
        # print("getAlphas()")
        if hasattr(self, "alphasParameters"):
            # print("hasttr")
            return self.alphasParameters
        #! why returning a set is correct
        self.alphasParameters = [
            v for k, v in self.named_parameters() if k=="alphas"
        ]
        #! I think it should return a list
        #! Module.register_parameter() takes iterable parameter, and set is a iterable
        # self.alphasParameters = [
        #     v for k, v in self.named_parameters() if k=="alphas"
        # ]
        print("\nalphas id in getAlphas()\n", id(self.alphas))
        return self.alphasParameters
    
    def getWeight(self):
        if hasattr(self, "weightParameters"):
            return self.weightParameters

        self.weightParameters = {
            v for k, v in self.named_parameters() if k!="alphas"
        }

        return self.weightParameters




if __name__ =="__main__":
    layer = Model(5, 2, 10)
    # print(list(layer.named_parameters()))
    tmp = {
        para for name, para in layer.named_parameters() if name == "alphas" 
    }
    for name, para in layer.named_parameters():

        if name == "alphas":
            pass
            # print("===", name)
            # tmp = {}
    # print(hasattr(layer, "_alphas"))
    # print(tmp)
    # print(type(layer.parameters()))
    for i in layer.parameters():
        pass
        # print(i)
    # print(layer)
    print(epoch_to_drop)
    for i in range(40):
        
        print(i in epoch_to_drop, i)