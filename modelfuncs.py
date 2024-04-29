import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from readbash import get_image_subregion_list, display_im

class ObjectDetectionCNN(nn.Module):
    def __init__(self):
        super(ObjectDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 4, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: 3 binary values for 3 objects
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = x.view(-1, 64 * 7 * 4)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation for binary output
        return x

#61 36

def pack_image_into_tensor(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im[0, 0] = np.min(im)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    im[0, 0] = 255
    image_tensor = torch.from_numpy(im).float()
    # Reshape the image tensor to have a channel dimension of size 1
    image_tensor = image_tensor.unsqueeze(0)

    # Add a batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def train_model():
    
    images = []
    labels = []

    M_image = 'images/training/M/Bash_M_resize.jpg'
    H_image = 'images/training/H/Bash_H_resize.jpg'
    empty_image = 'images/training/Empty/Bash_Empty_resize.jpg'

    for (yr, mth, im) in get_image_subregion_list(M_image):
        p = pack_image_into_tensor(im)
        label_tensor = torch.tensor([1, 0]).float()
        label_tensor = label_tensor.unsqueeze(0)
        images.append(p)
        labels.append(label_tensor)

    for (yr, mth, im) in get_image_subregion_list(H_image):
        p = pack_image_into_tensor(im)
        label_tensor = torch.tensor([0, 1]).float()
        label_tensor = label_tensor.unsqueeze(0)
        images.append(p)
        labels.append(label_tensor)

    for (yr, mth, im) in get_image_subregion_list(empty_image):
        p = pack_image_into_tensor(im)
        label_tensor = torch.tensor([0, 0]).float()
        label_tensor = label_tensor.unsqueeze(0)
        images.append(p)
        labels.append(label_tensor)

    batch_images = torch.cat(images, dim=0)
    batch_labels = torch.cat(labels, dim=0)


    train_data = images
    train_labels = labels

    # Create an instance of the model
    model = ObjectDetectionCNN()

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 1

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Testing
    #with torch.no_grad():
    #    outputs = model(test_data)
    #    predicted = (outputs > 0.5).float()  # Threshold the outputs to get binary predictions

    #print("Predicted object presence:")
    #print(predicted)

    torch.save(model.state_dict(),'models/diarymodel.pth')
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Pass the input through the model
        for im in images:
            output = model(im)
            # Check the outputs
            predicted_classes = (output > 0.5).float()
            print("Predicted classes:", predicted_classes)
