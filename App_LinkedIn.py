import streamlit as st 
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image


########### Read in data ##########
s = pd.read_csv("social_media_usage.csv")


def clean_sm(x):
    x = np.where(x == 1,1,0)
    return x

ss = pd.DataFrame({
    "income": np.where(s["income"] <= 9,s["income"],np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"]==1,1,0),
    "married": np.where(s["marital"]==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age": np.where(s["age"] <= 98, s["age"], np.nan),
    "sm_li": clean_sm(s["web1h"])  
})

ss = ss.dropna()


######### Train and Test ##########
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    random_state = 268)


########### Fit Model ############
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train) # fit algorithm to training data
y_pred = lr.predict(X_test) # make prediction on test set


###### Title Script #######
st.markdown("# Predicting LinkedIn User with Machine Learning")

linkedin_image  = Image.open("LinkedIn image.jpeg")
st.image(linkedin_image, use_column_width = True, caption = "Sourced from Google")

st.markdown("### This application is used for prediciting whether or not you are a LinkedIn user")
st.markdown("##### Please answer the following six questions to find out your prediction:")


####### input variables ##########
## income
income = st.selectbox(label = "What is your level of income?",
options = ("Less than $10k", 
            "$10k - $20k", 
            "$20k - $30k", 
            "$30k - $40k", 
            "$40k - $50k",
            "$50k - $75k", 
            "$75k - $100k", 
            "$100k - $150k", 
            "$150k or more"))

if income == "Less than $10k": income = 1
elif income == "$10k - $20k": income = 2
elif income == "$20k - $30k": income = 3
elif income == "$30k - $40k": income = 4
elif income == "$40k - $50k": income = 5
elif income == "$50k - $75k": income = 6
elif income == "$75k - $100k": income = 7
elif income == "$100k - $150k": income = 8
else: income = 9


## education
education = st.selectbox(label = "What is your education level?",
options = ("Less than Highschool", 
            "Highschool Incomplete", 
            "Highschool Graduate", 
            "Some College", 
            "Associate's Degree", 
            "Bachelor's Degree", 
            "Some Postgraduate",
            "Postgraduate Degree"))

if education == "Less than Highschool": education = 1
elif education == "Highschool Incomplete": education = 2
elif education == "Highschool Graduate": education = 3
elif education == "Some College": education = 4
elif education == "Associate's Degree": education = 5
elif education == "Bachelor's Degree": education = 6
elif education == "Some Postgraduate": education = 7
else: education = 8


## parent
parent = st.selectbox(label = "Are you a parent?",
options = ("Yes", "No"))
if parent == "Yes": parent = 1
else: parent = 0


## married
married = st.selectbox(label = "Are you married?",
options = ("Yes", "No"))
if married == "Yes": married = 1
else: married = 0
    

## female
female = st.selectbox("Do you identify as Male or Female?",
options = ("Male", "Female"))
if female == "Female": female = 1
else: female = 0


## age
age = st.number_input(label="What is your age?",
                min_value=1,
                max_value=100,
                value=1)

    
# probability
predict_user = [income, education, parent, married, female, age] 
probability = lr.predict_proba([predict_user]) #Find the probability
predicted_class = lr.predict([predict_user]) #Predict the class

if probability[0][1] >= 0.50:
    label = "Yes"
else:
    label = "No"

######### Add Button ########
if st.button("Predict"):
    predict_user = [income, education, parent, married, female, age] #Create data for example
    if predicted_class == 1: "You are predicted as a LinkedIn user"
    else: "You are predicted as not a LinkedIn user"
    probability = lr.predict_proba([predict_user]) #Find the probability
    st.write(f"Probability that you are a LinkedIn user: {probability[0][1]}")
    