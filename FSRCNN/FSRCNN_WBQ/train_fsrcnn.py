import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataSet_fsrcnn import DatasetFromFolder
from netModel_fsrcnn import FSRCNN
import os
from math import log10

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    batch_size = 1
    num_workers = 2
    zoom_factor = 2
    epoch_num=100
    train_path='F:\\Desktop\\All\\language\\VSCODEWORKSPACE\\Python\\Class\\计算机视觉\\SuperResolution\\SRCNN\\SRCNN-WBQ\\data\\train'
    test_path='F:\\Desktop\\All\\language\\VSCODEWORKSPACE\\Python\\Class\\计算机视觉\\SuperResolution\\SRCNN\\SRCNN-WBQ\\data\\test'
    model_path='./model/FSRCNN_model.pth'

    trainset = DatasetFromFolder(train_path,zoom_factor)
    testset = DatasetFromFolder(test_path,zoom_factor)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if os.path.exists(model_path):
        model = torch.load(model_path).to(device)
        print('加载先前模型成功')
    else:
        print('未加载原有模型训练')
        model = FSRCNN().to(device)


    lr=0.0005
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
    [
        {'params': model.first_part.parameters(), 'lr': lr },
        {'params': model.mid_part.parameters(), 'lr': lr },
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # 注意动态更改
    end_psnr=0
    for epoch in range(epoch_num):
        epoch_loss = 0
        for iteration, batch in enumerate(trainloader):
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")


        test_loss = 0
        avg_psnr = 0
        with torch.no_grad():
            for batch in testloader:
                input, target = batch[0].to(device), batch[1].to(device)

                out = model(input)
                loss = criterion(out, target)
                test_loss += loss.item()
                psnr = 10 * log10(1 / loss.item())
                avg_psnr += psnr
        ave_loss=test_loss / len(testloader)
        avg_psnr=avg_psnr / len(testloader)

        print(f"Epoch {epoch}. Testing loss: {ave_loss}")
        print(f"Average PSNR: {avg_psnr} dB.")

        if avg_psnr>end_psnr:
            torch.save(model, model_path)
            end_psnr=avg_psnr
            print('模型更新成功')
        print('*'*50)
    
        scheduler.step()
        # 打印当前学习率
        print('当前学习率为：',end=' ')
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        
        