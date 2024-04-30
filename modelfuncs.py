import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
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
    test_frac = 0.1
    batch_size = 32
    num_labels = 2

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


    #shuffle images and divide into test/train
    paired_lists = list(zip(images, labels))
    random.shuffle(paired_lists)
    images, labels = zip(*paired_lists)
    N_tot = len(images)
    N_test = math.floor(test_frac * N_tot)
    N_train = N_tot - N_test

    test_images = images[0:N_test]
    test_labels = labels[0:N_test]
    train_images = images[N_test:]
    train_labels = labels[N_test:]

    #training batches
    train_batches = []
    for i in range(0, len(train_images), batch_size):
        batch_images = torch.cat(train_images[i:i+batch_size], dim=0)
        batch_labels = torch.cat(train_labels[i:i+batch_size], dim=0)
        train_batches.append((batch_images, batch_labels))

    #testing batch
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Create an instance of the model
    model = ObjectDetectionCNN()

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        for (batch_images, batch_labels) in train_batches:
            # Forward pass
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(),'models/diarymodel.pth')

    # Testing
    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        predicted = (outputs > 0.5).float()  # Threshold the outputs to get binary predictions
        correct = (predicted == test_labels).float().sum()  # Count the number of correct predictions
        accuracy = correct * 100 / (num_labels * len(test_labels))  # Calculate the accuracy
        true_positives = ((predicted == 1) & (test_labels == 1)).sum().float()
        false_positives = ((predicted == 1) & (test_labels == 0)).sum().float()
        precision = 100 * true_positives / (true_positives + false_positives)
        print(f"Test Accuracy: {accuracy:.4f}%")
        print(f"Test Precision: {precision:.4f}%")
