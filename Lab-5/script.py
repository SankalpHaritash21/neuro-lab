import torch
import torch.nn as nn

# helper to count params
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_vgg_layers_small(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class SmallVGG(nn.Module):
    """
    Small VGG-like model for CIFAR (3x32x32).
    cfg example: [32,32,'M', 64,64,'M', 128,'M']
    """
    def __init__(self, num_classes=10, input_channels=3, batch_norm=False):
        super().__init__()
        # tiny configuration: few filters per block
        cfg = [32, 32, 'M', 64, 64, 'M', 128, 'M']
        self.features = make_vgg_layers_small(cfg, in_channels=input_channels, batch_norm=batch_norm)
        # reduce spatial to 1x1 so classifier is small
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # small classifier
        self.classifier = nn.Sequential(
            nn.Linear(128*1*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# Quick smoke test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallVGG(num_classes=10, input_channels=3, batch_norm=False).to(device)
    print("Parameters:", count_params(model))
    # Forward with CIFAR sized input
    x = torch.randn(4, 3, 32, 32, device=device)
    out = model(x)
    print("Output shape:", out.shape)  # expected (4,10)
