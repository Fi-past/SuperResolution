import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        scale_factor=2
        num_channels=1
        d=96
        s=24
        m=4
        self.first_part = nn.Sequential(
            # 1 56 224 224
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU()
        )
        # 56 12 224
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU()]
        for _ in range(m):
            # 12 12 224 224 224 224
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU()])
        # 12 56 224
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU()])
        self.mid_part = nn.Sequential(*self.mid_part)
        # 56 1 
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)
        
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x