# AI_task1
1. Train the Model using Teachable Machine:
Select "Image Project" in Teachable Machine.
Add at least two categories, such as "Tiger" and "Cheetah".
Upload images for each category.
After uploading the images, click the "Train" button to train the model.

3. Download the Model:
Once training is complete, click the "Export Model" button.
Choose the TensorFlow format to download the model in .json format.

4. Write a Python Script for Prediction:
Load the trained model using the TensorFlow library.
Load the new image you want to predict.
Convert the image to an array using image.load_img and image.img_to_array.
Normalize the image to be ready for model input.
Use the trained model to make a prediction with model.predict().
Select the category with the highest probability using np.argmax().
