# Instructions

- ## In this folder:

  - ### Suggested: create a "venv" first and activate it
  - run "pip install -r requirements.txt" to install all deps
  - run "python app.py" to start the backend server

- ## API:

  - Just like before you have to send the image to /predict endpoint
  - Also you can listen to webSocket for the "progress" event as written in the templates/index.html file
  - JSON Response is of the following format:
    - response
      - "predcicted_food_item" -> Predicted food label.
      - "ingredients" -> List of ingredients in the detected food item.
      - "risky_ingredients" -> Ingredients that we "don't" recomment to be eaten.
      - "risk_factor_percents" -> Percentage( how risky the food is ).
      - "nutritional_information": -> List( maybe long ) of the nutritional info of all the ingredients = - - present in the detected food item.
