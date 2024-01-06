from data import get_cumulative_nutrition, get_ingredient_matches
from ml import predict_image, get_risky_ingredients
import numpy as np
import cv2


# Generating the Cumulative json response
def process_image(img, med_sample, socketio):
    # Preprocessing the image -> (110, 110, 3) ...
    socketio.emit('progress', 'Processing Image...')
    test = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    test = test[:, :, :3]

    socketio.emit('progress', 'Normalizing Image...')
    # Normaling the image ...
    if np.any(test > 1):
        test = test / 255.

    # predicting the image label ...
    socketio.emit('progress', 'Predicting Image Label...')

    food_item = predict_image(test)

    # fetching the 'food_item' ingredients from DB.
    socketio.emit('progress', 'Getting Ingredient Info...')
    food_item_ingredients = get_ingredient_matches(food_item)

    # generating symptom recomendations / risk factors(in percents) ...
    socketio.emit('progress', 'Performing Risk Analysis...')
    risky_ingredients, risk_factor = get_risky_ingredients(
        med_sample, food_item_ingredients)

    # getting cumulative nutrition for all the ingedients present ...
    socketio.emit('progress', 'Getting Nutritional Info...')
    nutritional_info = get_cumulative_nutrition(food_item_ingredients)

    # formulating the json response ...
    response = {
        "predcicted_food_item": food_item,
        "ingredients": food_item_ingredients,
        "risky_ingredients": risky_ingredients,
        "risk_factor_percents": risk_factor,
        "nutritional_information": nutritional_info
    }

    socketio.emit('progress', 'Done')

    socketio.emit('progress', '')
    return response

# END
