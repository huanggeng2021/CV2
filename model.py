from torch import nn

# 分类任务
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input  [3, 28, 28]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # [32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # [32, 14, 14]

            nn.Conv2d(32, 16, 3, 1, 1),  # [32, 28, 28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [16, 7, 7]
        )
        self.fc = nn.Sequential(
            nn.Linear(16*7*7, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out = self.cnn(x)
        # print(out.size())   # [8, 16, 7, 7]
        out = out.view(out.size()[0], -1)
        return self.fc(out)