import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from netmodel_EDSR import EDSR
from dataset_EDSR import SRDataset
import os
from math import log10

if __name__ == '__main__':

    # train_path='G:/DIV2K/DIV2K/DIV2K_train_HR'
    # test_path='../../SRCNN/SRCNN-WBQ/data/Set5/set5'
    # fun    
    train_path='G:/SR/SRGAN/data/BSD100'
    test_path='F:/Desktop/All/language/VSCODEWORKSPACE/Python/Class/计算机视觉/SuperResolution/SRCNN/SRCNN-WBQ/data/Set5/set5'

    crop_size = 120      # 高分辨率图像裁剪尺寸
    scaling_factor = 2  # 放大比例

    # 学习参数
    checkpoint = '../model/EDSR.pth'   # 预训练模型路径，如果不存在则为None
    batch_size = 5    # 批大小
    start_epoch = 1     # 轮数起始位置
    epochs = 500        # 迭代轮数
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
        model = EDSR()

    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

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
        train_psnr=0
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
            psnr = 10 * log10(1 / loss.item())
            train_psnr+=psnr

        epoch_loss_train=train_loss / n_iter_train
        train_psnr=train_psnr/n_iter_train

        # tensorboard
        # writer.add_scalar('EDSR/MSE_Loss_train', epoch_loss_train, epoch)
        print(f"Epoch {epoch}. Training loss: {epoch_loss_train} Train psnr {train_psnr}DB")


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
                    writer.add_image('EDSR/epoch_'+str(epoch)+'_lr', make_grid(lr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                    writer.add_image('EDSR/epoch_'+str(epoch)+'_sr', make_grid(sr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)
                    writer.add_image('EDSR/epoch_'+str(epoch)+'_hr', make_grid(hr_imgs[:4,:3,:,:].cpu(), nrow=4, normalize=True),epoch)

        
        epoch_loss_test=test_loss/n_iter_test
        epoch_psnr=all_psnr / n_iter_test

        # tensorboard
        writer.add_scalars('EDSR/Loss', {
            'train_loss':epoch_loss_train,
            'test_loss':epoch_loss_test,
        },epoch)

        writer.add_scalars('EDSR/PSNR', {
            'train_psnr':train_psnr,
            'test_psnr':epoch_psnr,
        },epoch)

        print(f"Epoch {epoch}. Testing loss: {epoch_loss_test} Test psnr{epoch_psnr} dB")

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




