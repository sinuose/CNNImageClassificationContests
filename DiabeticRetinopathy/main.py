# Contest Submission for:
# https://www.kaggle.com/c/diabetic-retinopathy-detection/data

from hdr.CNN_model import *

# version: width, depth, res, dropout rate
efficient_net_config = {
"b0" : (1.0, 1.0, 224, 0.2),
"b1" : (1.0, 1.1, 240, 0.2),
"b2" : (1.1, 1.2, 260, 0.3),
"b3" : (1.2, 1.4, 300, 0.3),
"b4" : (1.4, 1.8, 380, 0.4),
"b5" : (1.6, 2.2, 456, 0.4),
"b6" : (1.8, 2.6, 528, 0.5),
"b7" : (2.0, 3.1, 600, 0.5)
}

if __name__ == "__main__":

    print("Program Start")
    # version readout
    version = 'b0'
    width_mult, depth_mult, res, dropout_rate = efficient_net_config[version]
    print(f"Running EfficientNet with params for Version: {version}")
    # generate version
    net = EfficientNet(width_mult, depth_mult, dropout_rate)
    if (1):
        x = torch.rand(1, 3, res, res)
        y = net(x)
        print(y.size())
