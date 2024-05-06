import torch
from readbash import get_image_subregion_list
from readbash import display_im
from modelfuncs import pack_image_into_tensor, ObjectDetectionCNN
import pandas as pd

loaded_model = ObjectDetectionCNN()

# Load the saved weights
loaded_model.load_state_dict(torch.load('models/diarymodel.pth'))


def get_diary_from_image(imagepath):
    #test_image = 'images/tests/mathytest1.jpg'
    test_image = imagepath
    #test_image = 'images/tests/bash_secondsheet_rotated.png'
    sr_list = get_image_subregion_list(test_image)

    loaded_model.eval()

    mths = ["Jan", "Feb", "Mar", "April", "May", "June"]
    df = pd.DataFrame(columns=[str(i) for i in range(1, 32)], dtype=str)

    for mth in mths:
        df.loc[mth] = [0] * 31

    for (mth, day, im) in sr_list:
        p = pack_image_into_tensor(im)
        with torch.no_grad():
            output = loaded_model(p)
            # Check the outputs
            predicted_classes = (output > 0.5).float()
            #print("Predicted classes:", predicted_classes)
            #display_im(im)
            first_element = predicted_classes[0, 0].item()
            second_element = predicted_classes[0, 1].item()
            classifier_label = " "
            if first_element == 1.0:
                classifier_label = "M"
            if second_element == 1.0:
                classifier_label = "H"
            df.loc[mth, str(day)] = classifier_label
    return df

