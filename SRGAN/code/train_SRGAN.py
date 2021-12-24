import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from netmodel_SRGAN import Generator, Discriminator, TruncatedVGG19
from dataset_SRGAN import SRDataset
import os
from math import log10

# # 数据集参数
# train_path='G:/SR/SRGAN/data/BSD100'
# test_path='F:/Desktop/All/language/VSCODEWORKSPACE/Python/Class/计算机视觉/SuperResolution/SRCNN/SRCNN-WBQ/data/Set5/set5'


if __name__ == '__main__':

    # 数据集参数
    train_path='G:/SR/SRGAN/data/BSD100'
    test_path='F:/Desktop/All/language/VSCODEWORKSPACE/Python/Class/计算机视觉/SuperResolution/SRCNN/SRCNN-WBQ/data/Set5/set5'

    crop_size = 96             # 高分辨率图像裁剪尺寸
    scaling_factor = 2         # 放大比例

    # 生成器模型参数(与SRResNet相同)
    large_kernel_size_g = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size_g = 3   # 中间层卷积的核大小
    n_channels_g = 64         # 中间层通道数
    n_blocks_g = 16           # 残差模块数量

    # 判别器模型参数
    kernel_size_d = 3  # 所有卷积模块的核大小
    n_channels_d = 64  # 第1层卷积模块的通道数, 后续每隔1个模块通道数翻倍
    n_blocks_d = 8     # 卷积模块数量
    fc_size_d = 1024   # 全连接层连接数

    # 学习参数
    batch_size = 10     # 批大小
    start_epoch = 0     # 迭代起始位置
    epochs = 10000         # 迭代轮数
    checkpoint = '../model/SRGAN.pth'   # 预训练模型路径，如果不存在则为None
    workers = 4         # 加载数据线程数量
    vgg19_i = 5         # VGG19网络第i个池化层
    vgg19_j = 4         # VGG19网络第j个卷积层
    beta = 1e-3         # 判别损失乘子
    vgg_loss_rate=1e-4
    lr = 1e-4           # 学习率

    # 多少步更新一次优化器 
    step_of_optimizer = 5
    # 每次优化器更新的程度
    optimizer_gamma=0.99

    max_psnr_test=25.23

    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ngpu = 1                 # 用来运行的gpu数量
    # writer = SummaryWriter('../log') # 实时监控     使用命令 tensorboard --logdir runs  进行查看



    # 模型初始化
    generator = Generator(large_kernel_size=large_kernel_size_g,
                                small_kernel_size=small_kernel_size_g,
                                n_channels=n_channels_g,
                                n_blocks=n_blocks_g,
                                scaling_factor=scaling_factor)

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                    n_channels=n_channels_d,
                                    n_blocks=n_blocks_d,
                                    fc_size=fc_size_d)

    # 初始化优化器
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad,generator.parameters()),lr=lr)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=step_of_optimizer, gamma=optimizer_gamma)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad,discriminator.parameters()),lr=lr)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=step_of_optimizer, gamma=optimizer_gamma)


    # 截断的VGG19网络用于计算损失函数
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # 损失函数
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # 将数据移至默认设备
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # 加载了多个模型
    if os.path.exists(checkpoint):
        print('加载原有模型成功')
        SRGAN_model = torch.load(checkpoint)
        # start_epoch = SRGAN_model['epoch'] + 1
        generator.load_state_dict(SRGAN_model['generator'])
        discriminator.load_state_dict(SRGAN_model['discriminator'])
        # optimizer_g.load_state_dict(SRGAN_model['optimizer_g'])
        # optimizer_d.load_state_dict(SRGAN_model['optimizer_d'])


    # 定制化的dataloaders
    train_dataset = SRDataset(train_path,
                            crop_size=crop_size,
                            scaling_factor=scaling_factor)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=workers) 

    test_dataset = SRDataset(test_path,
                            crop_size=crop_size,
                            scaling_factor=scaling_factor)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=workers) 




    # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):
        

        generator.train()   # 开启训练模式：允许使用批样本归一化
        discriminator.train()

        # 训练部分
        train_loss=0
        train_psnr=0
        test_loss=0
        test_psnr=0

        train_d_loss=0
        # 判别器
        train_d_acc_HR_num=0
        train_d_acc_SR_num=0
        train_d_acc_totalSum=0

        n_iter = len(train_loader)

        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 48, 48),  imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed 格式

            #-----------------------1. 生成器更新----------------------------

            # 生成
            sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), 范围在 [-1, 1]

            # 计算 VGG 特征图
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)              # batchsize X 512 X 6 X 6
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()     # batchsize X 512 X 6 X 6

            # 计算内容损失 mse vgg特征图上的
            content_loss = content_loss_criterion(sr_imgs_in_vgg_space,hr_imgs_in_vgg_space)

            # 原本的误差
            loss_sr_lr_mse = content_loss_criterion(sr_imgs,hr_imgs)

            # 计算生成损失
            sr_discriminated = discriminator(sr_imgs)  # (batch X 1)   
            adversarial_loss = adversarial_loss_criterion( #二分类
                sr_discriminated, torch.ones_like(sr_discriminated)) # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1

            # 计算总的感知损失 vgg特征图上的mse+生成损失(二元交叉熵)
            perceptual_loss = vgg_loss_rate*content_loss + beta * adversarial_loss+loss_sr_lr_mse

            # 后向传播.
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # 更新生成器参数
            optimizer_g.step()
            # 调整学习率
            scheduler_g.step()

            psnr = 10 * log10(1 / loss_sr_lr_mse)

            # 记录训练过程
            train_loss += perceptual_loss
            train_psnr += psnr


            #-----------------------2. 判别器更新----------------------------

            # 判别器判断
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            # sigmoid
            sigmoid = nn.Sigmoid()

            # batch_size是总数 计算准确率
            train_d_acc_totalSum+=batch_size
            prediction_hr = sigmoid(hr_discriminated)>0.5
            batchCorrect_hr = (prediction_hr == torch.ones_like(hr_discriminated)).sum().float()
            train_d_acc_HR_num+=batchCorrect_hr

            prediction_sr = sigmoid(sr_discriminated)<0.5
            batchCorrect_sr = (prediction_sr == torch.zeros_like(sr_discriminated)).sum().float()
            train_d_acc_SR_num+=batchCorrect_sr

            # 二值交叉熵损失
            # sr应该全判定为0，hr应该全部判定为1
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))  # 判别器希望能够准确的判断真假，因此凡是生成器生成的都设置为0，原始图像均设置为1

            # 后向传播
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # 更新判别器
            optimizer_d.step()
            scheduler_d.step()

            # 记录损失
            train_d_loss=adversarial_loss


        train_loss/=n_iter
        train_psnr/=n_iter


        # ----------------------------测试-------------------------------
        with torch.no_grad():

            generator.eval()   # 开启训练模式：允许使用批样本归一化
            discriminator.eval()

            n_iter = len(test_loader)

            # 按批处理
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):

                # 数据移至默认设备进行训练
                lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 48, 48),  imagenet-normed 格式
                hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed 格式

                #-----------------------1. 生成器更新----------------------------

                # 生成
                sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), 范围在 [-1, 1]

                # 计算 VGG 特征图
                sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)              # batchsize X 512 X 6 X 6
                hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()     # batchsize X 512 X 6 X 6

                # 计算内容损失 mse vgg特征图上的
                content_loss = content_loss_criterion(sr_imgs_in_vgg_space,hr_imgs_in_vgg_space)

                # 计算生成损失
                sr_discriminated = discriminator(sr_imgs)  # (batch X 1)   
                adversarial_loss = adversarial_loss_criterion( #二分类
                    sr_discriminated, torch.ones_like(sr_discriminated)) # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1

                # 原本的误差
                loss_sr_lr_mse = content_loss_criterion(sr_imgs,hr_imgs)

                # 计算总的感知损失 vgg特征图上的mse+生成损失(二元交叉熵)
                perceptual_loss = content_loss + beta * adversarial_loss+loss_sr_lr_mse


                psnr = 10 * log10(1 / loss_sr_lr_mse)

                # 记录训练过程
                test_loss+=perceptual_loss
                test_psnr+=psnr


                # # 监控图像变化
                # if i==(n_iter-1):
                #     writer.add_image('SRGAN/epoch_'+str(epoch)+'_1', make_grid(lr_imgs[:4,:,:,:].cpu(), nrow=4, normalize=True),epoch)
                #     writer.add_image('SRGAN/epoch_'+str(epoch)+'_2', make_grid(sr_imgs[:4,:,:,:].cpu(), nrow=4, normalize=True),epoch)
                #     writer.add_image('SRGAN/epoch_'+str(epoch)+'_3', make_grid(hr_imgs[:4,:,:,:].cpu(), nrow=4, normalize=True),epoch)


            test_loss/=n_iter
            test_psnr/=n_iter
        

            # 如果psnr提升，则更新
            if test_psnr>max_psnr_test:
                print('更新成功')
                torch.save({
                    # 'epoch': epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    # 'optimizer_g': optimizer_g.state_dict(),
                    # 'optimizer_d': optimizer_d.state_dict(),
                }, checkpoint)
                max_psnr_test=test_psnr

            # print(max_psnr_test)

            # # 监控损失值变化 生成器
            # writer.add_scalars('SRGAN/loss_g', {
            #     'train_loss_g':train_loss,
            #     'test_loss_g':test_loss,
            # },epoch)

            # writer.add_scalars('SRGAN/psnr', {
            #     'train_psnr':train_psnr,
            #     'test_psnr':test_psnr,
            # },epoch)

            # writer.add_scalars('SRGAN/loss_d_train', {
            #     'train_loss_d':train_d_loss,
            # },epoch)

            # writer.add_scalars('SRGAN/loss_d_train_acc', {
            #     # 越高越好
            #     'train_d_hr_acc':train_d_acc_HR_num/train_d_acc_totalSum,
            #     # 越低越好
            #     'train_d_sr_acc':train_d_acc_SR_num/train_d_acc_totalSum,
            # },epoch)

            print(f"Epoch {epoch} Training loss: {train_loss} Train psnr {train_psnr}DB")
            print(f"Epoch {epoch} Testing loss: {test_loss} Test psnr{test_psnr}DB")
            print(f"Epoch {epoch} d_hr_acc: {train_d_acc_HR_num/train_d_acc_totalSum} d_sr_acc{train_d_acc_SR_num/train_d_acc_totalSum} loss_d {train_d_loss}")


    # 训练结束关闭监控
    # writer.close()