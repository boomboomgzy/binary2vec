from param_parser import parameter_parser
from binary2vec import Binary2vecTrainer,predit
import torch
import torch.multiprocessing as mp
import os
import random
import numpy as np

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()

    random.seed(18)
    np.random.seed(18)
    torch.manual_seed(18)
    torch.cuda.manual_seed_all(18)
    torch.cuda.set_device(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#训练
#    trainer = Binary2vecTrainer(args)
#    trainer.fit()
#    metric=trainer.score(mode='test')
#    print(f'test metric: {str(round(metric, 10))}')

#测试
#    trainer = Binary2vecTrainer(args)
#    metric=trainer.score(mode='test')
#    print(f'test metric: {str(round(metric, 10))}')

#推理
#    predit_homoG_dir=r''
#    predit_result_dir=r''
#    predit(args,predit_homoG_dir,predit_result_dir)

if __name__ == "__main__":
    main()
