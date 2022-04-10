0409
resolve 0405 experiment questions.
training 5 rounds with the same initail weight.
conclusion:
I got the same same nas mode and same trained model. 
Without a doubt, testing accuracy is all the same.



0405
two parallel innercell, five layers
conclusion:
It's strange that the same initail weight will get different model and result accuracy. Please make sure there is nothing wrong during training proceudre.




3/10
pick the operation in a cell according to two top-2 alphas
## procedure
1. commnad "python train_nas_5cell.py"
train_nas_5cell.py is a more readable file modified from train_search_5cell.py
2.  command "python decode_pdarts.py"
用pickSecondMax()來選出 ./alpha_pdart_nodrop內的第二大alphas
3. command "python retrain_5cell.py"
4. command "python test.py"


