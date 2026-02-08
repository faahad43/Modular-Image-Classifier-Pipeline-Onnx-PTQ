from torchvision.models import convnext_tiny, vit_b_32, ConvNeXt_Tiny_Weights, ViT_B_32_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch import nn
import torch.nn.functional as F

class FasterRCNNBackboneClassifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone):
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # assuming FPN outputs 256 channels (default for torchvision)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # FasterRCNN backbone returns an OrderedDict of feature maps
        features = self.backbone(x)
        # pick one feature map, usually '0' or the highest level
        feat = features['0']
        pooled = self.global_pool(feat)
        pooled = pooled.view(pooled.size(0), -1)  # flatten
        out = self.classifier(pooled)
        return out

class customCNN(nn.Module):
    def __init__(self, num_classes):
        super(self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def make_model(num_classes, arch, pretrained=True, freeze_backbone=True):
    if arch == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights= weights)
    
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    
    elif arch == 'vit_b_32':
        weights = ViT_B_32_Weights.DEFAULT if pretrained else None
        model = vit_b_32(weights= weights)
        
        # the VIT has three parts conv_proj, encoder and head, on freezing we will freeze first 2
        if freeze_backbone:
            for param in model.conv_proj.parameters():
                param.requires_grad = False
            for param in model.encoder.parameters():
                param.requires_grad = False
        
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, num_classes)
        return model
        
    elif arch == 'faster_rcnn':
        weights_backbone = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        backbone = fasterrcnn_resnet50_fpn(weights=None, weights_backbone = weights_backbone).backbone
        
        model = FasterRCNNBackboneClassifier(backbone, num_classes, freeze_backbone)
        return model
    
    elif arch == 'custom_cnn':
        model = customCNN(num_classes)
        return model
            
    raise ValueError(f"Unsupported model architecture: {arch}")