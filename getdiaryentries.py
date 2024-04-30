import torch
from readbash import get_image_subregion_list
from readbash import display_im
from modelfuncs import pack_image_into_tensor, ObjectDetectionCNN

loaded_model = ObjectDetectionCNN()

# Load the saved weights
loaded_model.load_state_dict(torch.load('models/diarymodel.pth'))

test_image = 'images/tests/mathytest1.jpg'
#test_image = 'images/tests/bash_secondsheet_rotated.png'
sr_list = get_image_subregion_list(test_image)

loaded_model.eval()

for (yr, mth, im) in sr_list:
    p = pack_image_into_tensor(im)
    with torch.no_grad():
        output = loaded_model(p)
        # Check the outputs
        predicted_classes = (output > 0.5).float()
        #print("Predicted classes:", predicted_classes)
        #display_im(im)
        first_element = predicted_classes[0, 0].item()
        second_element = predicted_classes[0, 1].item()
        if first_element == 1.0:
            print("Migraine!")
        if second_element == 1.0:
            print("Headache!")

