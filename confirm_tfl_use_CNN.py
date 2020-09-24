from PIL import Image, ImageOps
import numpy as np

#========================Auxiliary Function===========================

def crop(image, coordinate):
    image = Image.open(image)
    image = ImageOps.expand(image, border=41, fill='black')
    im = image.crop((round(coordinate[0]) , round(coordinate[1]) , round(coordinate[0]) + 81, round(coordinate[1]) + 81))
    return im

def is_traffic_light(model, image):
    image = np.array(image)
    crop_shape = (81, 81)
    test_image = image.reshape([-1] + list(crop_shape) + [3])
    predictions = model.predict(test_image)
    predicted_label = np.argmax(predictions, axis=-1)
    if predicted_label[0] == 1:
        return True
    return False

#========================End of Auxiliary Functions===========================


def confirm_tfl_by_CNN(curr_frame, model):

    for i, dot in enumerate(curr_frame.cordinates):
        im = crop(curr_frame.img, dot)
        predicted_label = is_traffic_light(model, im)
        if predicted_label:
            curr_frame.traffic_light.append(dot)
            curr_frame.colors.append(curr_frame.cordinates_colors[i])

    curr_frame.traffic_light = np.array(curr_frame.traffic_light)