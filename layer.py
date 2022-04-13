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
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        #info two innerCells per layer 
        self.innerCellList = nn.ModuleDict({
            'innerCell_'+str(layer)+'_0': cell(inputChannel, outputChannel, stride),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel, stride),
        })
    def forward(self, input, alphas):

        indexOfInnerCell = 0
        output = 0
        for name in self.innerCellList:
            # add each inner cell directly without alphas involved
            output = output + self.innerCellList[name](input, alphas[indexOfInnerCell])
            indexOfInnerCell = indexOfInnerCell + 1
            print("innerCellList{} output".fomrat(name), output)

        return output
    
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
    

    