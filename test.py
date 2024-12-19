from model.main import *

if __name__ == '__main__':
    save_path = r'D:/PyCharm/PyCode/Hemangioma/Data/train_dataset/test/result'
    model_path = r'checkpoints/best.pth'
    dataset_test = Data('D:/PyCharm/PyCode/Hemangioma/Data/train_dataset/test', scale=(DefaultConfig.crop_width,
                                                                         DefaultConfig.crop_height), mode='test')
    args = DefaultConfig()
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = UNet(n_channels=3, n_classes=2)
    cudnn.benchmark = True
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    test(model, dataloader_test, args, save_path)