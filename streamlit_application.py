import streamlit as st
import pickle

with open('my_model.pkl','rb') as file:
    XG_model=pickle.load(file)

# Define the Streamlit app
def app():
    st.title("Vine Quality Prediction")
    st.write(
        "This application uses all features of wine to predict if wine have good quality or bad")

    # Create input widgets for the user to enter the characteristics of an iris flower
    fixed_acidity = st.slider("Fixed acidity :", 0.0, 15.0, 0.1)
    volatile_acidity = st.slider("Volatile acidity :", .0, 1.5, 0.001)
    citric_acid = st.slider("Citric acid :", 0.0, 1.0, 0.01)
    residual_sugar = st.slider("Residual sugar :", 0.0, 15.0, 0.1)
    chlorides = st.slider("Chlorides :", 0.0, 0.6, 0.001)
    free_sulfur = st.slider("Free sulfur dioxide :", 0.0, 80.00, 1.0)
    total_sulfur = st.slider("Total sulfur dioxide :", 0.0, 300.0, 1.0)
    density = st.slider("Density :", 0.0, 2.0, 0.01)
    ph = st.slider("pH :", 0.0, 4.0, 0.01)
    sulfates = st.slider("Sulfates :", 0.0, 2.0, 0.01)
    alcohol = st.slider("Alcohol :", 0.0, 15.0, 0.1)

    clicked=st.button("Submit")

    # Make a prediction based on the user's input
    if clicked:
        prediction = XG_model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur, total_sulfur, density, ph, sulfates,
                                    alcohol]])

        # Show the predicted species of the iris flower
        if prediction == 1:
            st.balloons()
            st.success('Good quality')
        else:
            st.snow()
            st.info('Not good quality')


# Run the Streamlit app
if __name__ == '__main__':
    app()