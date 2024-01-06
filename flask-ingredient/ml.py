import numpy as np
from tensorflow import keras
from data import symp_dec_labels, symp_dec_data, image_labels
from sklearn.cluster import KMeans

# # columns = ['Digestive discomfort', 'Dairy Allergies', 'Allergies', 'Nut allergies', 'Gas', 'Bloating', 'Blood sugar',
# #            'Dental issues', 'Foodborne illnesses', 'cholesterol', 'Lactose intolerance',
# #            'Gluten sensitivity/intolerance', 'Weight gain', 'Heart issues']

# symptom deCoder Model...
decoder_model = KMeans(n_clusters=15, random_state=0).fit(symp_dec_data)

clustered = {}

for i, row in enumerate(symp_dec_data):
    cat = decoder_model.predict(np.expand_dims(row, 0)).tolist()[0]
    if clustered.get(str(cat), None) == None:
        clustered[str(cat)] = [symp_dec_labels[i]]
    else:
        clustered[str(cat)].append(symp_dec_labels[i])


def get_risky_ingredients(sample, ingredients):
    matching_risky_ingredients = []
    risky_ingredients = []
    risk_factor = 0

    for idx in np.where(sample == 1)[0]:
        sample_ = np.zeros_like(sample)
        sample_[idx] = 1
        cluster = decoder_model.predict(np.expand_dims(sample_, 0))
        risky_ingredients.extend(clustered.get(str(cluster[0])))

    if not np.all(sample == 0):
        cluster = decoder_model.predict(np.expand_dims(sample, 0))
        risky_ingredients.extend(clustered.get(str(cluster[0])))

    for ingredient_ in ingredients:
        if ingredient_ in risky_ingredients:
            matching_risky_ingredients.append(ingredient_)
            risk_factor += 1

    risk_factor = round(
        (risk_factor / (len(risky_ingredients)+1e-16)) * 100.0, 1)
    return (matching_risky_ingredients, risk_factor)


# loading the ImageRecog model
img_model = keras.models.load_model('models/my_food_model_99pct.h5')


def predict_image(img):
    pred = img_model.predict(np.expand_dims(img, 0))
    pred_index = np.argmax(pred)
    image_class = image_labels[pred_index]
    return image_class

# END

# For Testing ...
# sample = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
