from numpy import dtype
import torch
import torch.nn as nn
from models.alpha.cell import Cell_conv
#info Cell_conv actually is innerCell
class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, inputChannel=3, outputChannel=96, stride=1, padding=1,  cell=Cell_conv):
        super(Layer, self).__init__()
        self.numOfInnerCell = numOfInnerCell
        self.layer = layer
        # self.initializeAlphas()
        self.minAlpha1 = True
        self.minAlpha2 = True
        self.minAlpha3 = True
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        # self.opList = nn.ModuleDict({
        #     'conv_'+str(layer)+'_1': cell(3, 96, 4),
        #     'max_pool_'+str(layer)+'_1': nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     'conv_'+str(layer)+'_2': cell(96, 256, 1),
        #     'max_pool_'+str(layer)+'_2': nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     'conv_'+str(layer)+'_3': cell(256, 384, 1),
        #     'conv_'+str(layer)+'_4': cell(384, 384, 1),
        #     'conv_'+str(layer)+'_5': cell(384, 256, 1),
        #     'max_pool_'+str(layer)+'_3': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # })
        #info two innerCells per layer 
        self.opList = nn.ModuleDict({
            'innerCell_'+str(layer)+'_0': cell(inputChannel, outputChannel, stride),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel, stride),
        })
    def forward(self, input, alphas):
        indexOfInnerCell = 0
        output = 0
        for name in self.opList:
            # add each inner cell directly without alphas involved
            output = output + self.opList[name](input, alphas[indexOfInnerCell])
            indexOfInnerCell = indexOfInnerCell + 1
        return output
    
    def test(self):
        print("length", len(self.opList))
        index = 0
        for name in self.opList:
            print(index, name,  self.opList[name])
            index = index + 1
    
class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
            # nn.BatchNorm2d(C_out, affine=affine),
            # nn.ReLU(inplace=False),
        )
        self._initialize_weights() #* initialize kernel weights

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
if __name__ == '__main__':
    layer = Layer(5, 0, 3, 2, 1)
    # model.test()
    # print(model)
    input = torch.ones((1, 3, 12, 12), dtype=torch.float).requires_grad_()
    alphas = torch.ones((2, 5), dtype=torch.float)
    # print(input)
    sum = 5
    # sum = sum + input[0] + input[1]
    # sum.backward()
    # print(2*input[0])
    # print(layer(input, alphas))

    out = layer(input, alphas)
    print(out.shape)
    print(out)
    

    