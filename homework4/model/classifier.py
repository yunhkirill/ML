import torch
import torch.nn as nn

    
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, input_shape=(3, 32, 32)):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.encoder.eval()
        
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            features = self.encoder(dummy_input)
            feature_dim = features.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits