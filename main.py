import torch
import torchvision.models as models
import torchvision.transforms as transforms

class StyleTransferEngine:
    """
    Neural Style Transfer Engine
    Maintains custom VGG loss optimization networks to transfer style textures to content images.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()

    def extract_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

if __name__ == "__main__":
    engine = StyleTransferEngine()
    print(f"Style Transfer VGG Feature Extractor initialized successfully on device: {engine.device}!")
