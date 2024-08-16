import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model2.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("🍎🍇FRUITS & VEGETABLES RECOGNITION SYSTEM 🥕🥬")
    image_path = "home_img.jpg"
    st.image(image_path)

#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.markdown("""
    **Fruits**: 
    - Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
    
    **Vegetables**: 
    - Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chili Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant
    """)
    st.subheader("Content")
    st.markdown("""
    - **Train**: 100 images each
    - **Test**: 10 images each
    - **Validation**: 10 images each
    """)

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", width=400)

        #Predict button
        if(st.button("Predict")):
            with st.spinner("Predicting..."):
                result_index = model_prediction(test_image)
                # Reading Labels
                with open("labels.txt") as f:
                    labels = [line.strip() for line in f]
                st.success(f"Model is predicting it's a **{labels[result_index]}**!")
                st.balloons()
