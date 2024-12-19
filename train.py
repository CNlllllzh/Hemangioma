from model.main import *

if __name__ == '__main__':
    seed=1234
    torch.manual_seed(seed)   # 固定初始化
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()
    modes = 'train'
    if modes=='train':
        main(mode='train', args=args)