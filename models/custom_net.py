from torch import nn

# AIM: classification of the image
# Define the custom neural network
class CustomNet(nn.Module): # base class of the Neural Model in PyTorch
    def __init__(self): # Operation that I want the model does
        super(CustomNet, self).__init__()

        # Define layers of the neural network - 5 different in that occasion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2) # 64 is the output of the first layer, that are the nodes...
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # ... that become the input of the next layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)

        self.flatten = nn.Flatten() # Flatten the output of the previous layer

        # The lass step that we have to perform
        self.fc1 = nn.Linear(25088, 200) # 200 is the number of classes in TinyImageNet, you have to chose here a value (256, 200)


    def forward(self, x): # Call and assigned the operation (written in init)
        # Define forward pass
        x = self.conv1(x).relu() # Convolutional layer, the attivation function ReLU
        # print(f"x.shape for cov1: {x.shape}")
        x = self.conv2(x).relu()
        # print(f"x.shape for cov2: {x.shape}")
        x = self.conv3(x).relu()
        # print(f"x.shape for cov3: {x.shape}")
        x = self.conv4(x).relu()
        # print(f"x.shape for cov4: {x.shape}")

        # x = self.conv5(x).relu()
        # print(f"x.shape for cov5: {x.shape}")

        x = self.flatten(x)
        # print(f"x.shape for flatten: {x.shape}")
        x = self.fc1(x)
        # print(f"x.shape for fc1: {x.shape}")
        
        return x