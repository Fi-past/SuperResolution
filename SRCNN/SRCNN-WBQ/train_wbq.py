import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataSet_wbq import DatasetFromFolder
from netModel_wbq import SRCNN
import os
from math import log10

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    batch_size = 1
    num_workers = 2
    zoom_factor = 2
    crop_size = 224
    epoch_num=200
    train_path='./data/train'
    test_path='./data/test'
    model_path='./model/SRCNN_model.pth'

    trainset = DatasetFromFolder(train_path,zoom_factor,crop_size)
    testset = DatasetFromFolder(test_path,zoom_factor,crop_size)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if os.path.exists(model_path):
        model = torch.load(model_path).to(device)
        print('加载先前模型成功')
    else:
        print('未加载原有模型训练')
        model = SRCNN().to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr": 0.001},  
            {"params": model.conv2.parameters(), "lr": 0.001},
            {"params": model.conv3.parameters(), "lr": 0.0001},
        ], lr=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 注意动态更改
    pre_loss=1000
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

        print(f"Epoch {epoch}. Testing loss: {ave_loss}")
        print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

        if ave_loss<pre_loss:
            torch.save(model, model_path)
            pre_loss=ave_loss
            print('模型更新成功')
        print('*'*50)

        scheduler.step()
        # 打印当前学习率
        print('当前学习率为：',end=' ')
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    