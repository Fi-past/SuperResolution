import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from netmodel_resnet import SRResNet
from dataset_resnet import SRDataset
import os
from math import log10

if __name__ == '__main__':
    # G:/DIV2K/DIV2K/DIV2K_train_HR
    # train_path='../../SRCNN/SRCNN-WBQ/data/train'
    train_path='G:/DIV2K/DIV2K/DIV2K_train_HR'
    test_path='../../SRCNN/SRCNN-WBQ/data/Set5/set5'

    crop_size = 224      # 高分辨率图像裁剪尺寸
    scaling_factor = 2  # 放大比例

    # 模型参数
    large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size = 3   # 中间层卷积的核大小
    n_channels = 64         # 中间层通道数
    n_blocks = 16           # 残差模块数量

    # 学习参数
    checkpoint = '../model/rsresnet.pth'   # 预训练模型路径，如果不存在则为None
    batch_size = 8    # 批大小
    start_epoch = 1     # 轮数起始位置
    epochs = 100        # 迭代轮数
    workers = 2         # 工作线程数
    lr = 1e-4           # 学习率

    # 先前的psnr
    pre_psnr=0

    # 设备参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    ngpu = 1

    # cudnn.benchmark = True # 对卷积进行加速

    writer = SummaryWriter('../log') # 实时监控     使用命令 tensorboard --logdir runs  进行查看


    if os.path.exists(checkpoint):
        model = torch.load(checkpoint).to(device)
        print('加载先前模型成功')
    else:
        print('未加载原有模型训练')
        model = SRResNet(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)

    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(train_path, crop_size, scaling_factor)
    test_dataset = SRDataset(test_path, crop_size, scaling_factor)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

        # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):

        model.train()  # 训练模式：允许使用批样本归一化
        train_loss=0
        n_iter_train = len(train_loader)

        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        epoch_loss_train=train_loss / n_iter_train

        # tensorboard
        writer.add_scalar('SRResNet/MSE_Loss_train', epoch_loss_train, epoch)
        print(f"Epoch {epoch}. Training loss: {epoch_loss_train}")



        model.eval()  # 测试模式
        test_loss=0
        all_psnr = 0
        n_iter_test = len(test_loader)
        # print(n_iter)
        # print(len(test_dataset))
        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)

                psnr = 10 * log10(1 / loss.item())
                all_psnr+=psnr
                test_loss+=loss.item()

                # 可视化图像
                if i==n_iter_test-1:
                    writer.add_image('SRResNet/epoch_'+str(epoch)+'_lr', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                    writer.add_image('SRResNet/epoch_'+str(epoch)+'_sr', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                    writer.add_image('SRResNet/epoch_'+str(epoch)+'_hr', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)

        
        epoch_loss_test=test_loss/n_iter_test
        epoch_psnr=all_psnr / n_iter_test

        # tensorboard
        writer.add_scalar('SRResNet/MSE_Loss_test', epoch_loss_test, epoch)
        writer.add_scalar('SRResNet/test_psnr', epoch_psnr, epoch)



        print(f"Epoch {epoch}. Testing loss: {epoch_loss_test}")
        print(f"Average PSNR: {epoch_psnr} dB.")

        if epoch_psnr>pre_psnr:
            torch.save(model, checkpoint)
            pre_psnr=epoch_psnr
            print('模型更新成功')
        
        scheduler.step()
        # 打印当前学习率
        print('当前学习率为：',end=' ')
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('*'*50)
        
    # 训练结束关闭监控
    writer.close()




