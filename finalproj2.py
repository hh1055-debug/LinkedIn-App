import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report
)
import streamlit as st
import altair as alt


#######################################################

#Put in all of the code to clean and set up the data
s= pd.read_csv("social_media_usage.csv")

def clean_sm (x):
  x= np.where(x==1, 1, 0)
  return x


ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": s["income"],
    "education": s["educ2"],
    "parent": s["par"],
    "married": s["marital"],
    "female": s["gender"],
    "age": s["age"]
})


#make any income values greater than 9 NA
ss.loc[ss["income"] > 9, "income"]= np.nan

#check to see if any >9 values remain
print(sum(ss["income"] > 9))

#make any educ values greater than 8 NA
ss.loc[ss["education"] > 8, "education"]= np.nan

#check to see if any >8 values remain
print(sum(ss["education"] > 8))

#make parent a binary, 1= yes, 0 = no
ss["parent"]= clean_sm(ss["parent"])

#Check there are no values > 1 remaining
print(sum(ss["parent"] > 1))

#make married a binary, 1= yes, 0 = no
ss["married"]= clean_sm(ss["married"])

#Check there are no values > 1 remaining
print(sum(ss["married"] > 1))


#Make female =1, right now it's coded as 2, with male as 1.
ss["female"]= np.where(ss["female"]==2, 1, 0)

#check to make sure all values are 0 or 1
print(sum(ss["female"] > 1))

#make any age values greater than 98 NA
ss.loc[ss["age"] > 98, "age"]= np.nan
print(sum(ss["age"] > 98))

#check for missing values
print(ss.isna().sum())

#drop the NAs
ss = ss.dropna()

#check to make sure no more NAs
print(ss.isna().sum())

#transform the binary variables into integers
ss = ss.astype({"sm_li": int, "income": int, "education": int, "age": int})

ss.describe()

#pairs plot
axes = pd.plotting.scatter_matrix(
    ss[["sm_li", "income", "education", "parent", "married", "female", "age"]],
    figsize=(9, 9),
    diagonal='hist',
    alpha=0.6
)
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(0)
    ax.yaxis.label.set_rotation(45)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.show()

#Correlation matrix
ss[["sm_li", "income", "education", "parent", "married", "female", "age"]].corr().round(2)

sm_li_1 = ss[ss["sm_li"] == 1]
sm_li_0 = ss[ss["sm_li"] == 0]

#show summary metrics for LinkedIn users
sm_li_1.describe()

#show summary metrics for non-LinkedIn users
sm_li_0.describe()

colors = ["#E07B39", "#0A66C2"]
labels = ["Non-LinkedIn", "LinkedIn"]

fig, axes = plt.subplots(3, 2, figsize=(12, 13))

pd.crosstab(ss["income"], ss["sm_li"], normalize="columns").plot(
    kind="bar", ax=axes[0, 0], color=colors, rot=0, legend=False)
axes[0, 0].set_title("Income")
axes[0, 0].set_ylabel("Share of group")

pd.crosstab(ss["education"], ss["sm_li"], normalize="columns").plot(
    kind="bar", ax=axes[0, 1], color=colors, rot=0, legend=False)
axes[0, 1].set_title("Education")
axes[0, 1].set_ylabel("Share of group")

pd.crosstab(ss["parent"], ss["sm_li"], normalize="columns").plot(
    kind="bar", ax=axes[1, 0], color=colors, rot=0, legend=False)
axes[1, 0].set_title("Parent")
axes[1, 0].set_ylabel("Share of group")

pd.crosstab(ss["married"], ss["sm_li"], normalize="columns").plot(
    kind="bar", ax=axes[1, 1], color=colors, rot=0, legend=False)
axes[1, 1].set_title("Married")
axes[1, 1].set_ylabel("Share of group")

pd.crosstab(ss["female"], ss["sm_li"], normalize="columns").plot(
    kind="bar", ax=axes[2, 0], color=colors, rot=0, legend=False)
axes[2, 0].set_title("Female")
axes[2, 0].set_ylabel("Share of group")

axes[2, 1].hist(ss.loc[ss["sm_li"] == 0, "age"], bins=20, density=True, alpha=0.5, color=colors[0])
axes[2, 1].hist(ss.loc[ss["sm_li"] == 1, "age"], bins=20, density=True, alpha=0.5, color=colors[1])
axes[2, 1].set_title("Age")
axes[2, 1].set_ylabel("Density")
axes[2, 1].set_xlabel("age")

axes[0, 0].legend(labels, title="")
plt.tight_layout()
plt.show()

prop = pd.crosstab(ss["income"].astype(int), ss["sm_li"], normalize="columns")
prop.columns = ["Non-LinkedIn", "LinkedIn"]

prop.plot(kind="bar", color=["#E07B39", "#0A66C2"], rot=0)
plt.xlabel("Income level")
plt.ylabel("Share of group")
plt.title("Income distribution by LinkedIn use")
plt.legend(title="")
plt.show()

prop = pd.crosstab(ss["female"].astype(int), ss["sm_li"], normalize="columns")
prop.columns = ["Non-LinkedIn", "LinkedIn"]

prop.plot(kind="bar", color=["#E07B39", "#0A66C2"], rot=0)
plt.xlabel("Female")
plt.ylabel("Share of group")
plt.title("Gender distribution by LinkedIn use")
plt.legend(title="")
plt.show()


y= ss["sm_li"]
x= ss[["income", "education", "parent", "married", "female", "age"]]

y.head()

x.head()

## Q5


#Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state= 100)

print(x_train.shape, x_test.shape)

model = LogisticRegression(class_weight="balanced")

model.fit(x_train, y_train)

print(f"The intercept is", model.intercept_)
print(f"The Variable coefficients are",model.coef_)

model.get_params()




test_probs = model.predict_proba(x_test)[:, 1]

test_pred_class = model.predict(x_test)

print("The accuracy for this model on the test set is:", round(accuracy_score(y_test, test_pred_class), 3))

print("Confusion matrix (threshold = 0.5):")
print(pd.DataFrame(
    confusion_matrix(y_test, test_pred_class)
))





conf_matrix = pd.DataFrame(
    confusion_matrix(y_test, test_pred_class),
    index=["Actually Do Not Use LinkedIn", "Actually Use LinkedIn"],
    columns=["Pred Don't Use LinkedIn", "Pred Use LinkedIn"],
)
print(conf_matrix)


#calculating by hand

pred_true = (test_pred_class == 1)
pred_false = (test_pred_class == 0)

ac_true = (y_test == 1)
ac_false = (y_test == 0)


tp = sum(pred_true & ac_true)
fp = sum(pred_true & ac_false)
tn = sum(pred_false & ac_false)
fn = sum(pred_false & ac_true)


print(tp,fp,tn,fn)

manual_prec = tp / (tp + fp)
manual_rec = tp / (tp + fn)
manual_f1 = 2 * ((manual_prec * manual_rec)/ (manual_prec + manual_rec))

print(f"Precision is:", manual_prec)
print(f"Recall is", manual_rec)
print(f"F1 is", manual_f1)

print(classification_report(y_test, test_pred_class))


ss.head()

person1 = pd.DataFrame([[8, 7, 0, 1, 1, 42]], columns=x.columns)

# Predict class, given input features
predicted_class = model.predict(person1)

# Generate probability of positive class (=1)
probs = model.predict_proba(person1)

# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}")
print(f"Probability that this person is a LinkedIn User: {probs[0][1]}")

#Change the age to 82, then predict again
person2 = pd.DataFrame([[8, 7, 0, 1, 1, 82]], columns=x.columns)

# Predict class, given input features
predicted_class = model.predict(person2)

# Generate probability of positive class (=1)
probs = model.predict_proba(person2)

# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}")
print(f"Probability that this person is a LinkedIn User: {probs[0][1]}")



####################################################


#Begin with Streamlit specific code

st.markdown("# Group 5 Streamlit Application")

######################################################
#Begin Summary Metrics
st.markdown("## Dashboard Metrics")

if st.checkbox("Show Summary Metrics"):
    st.write(ss.describe())

######################################################
#Begin charts
st.markdown("## Dashboard Graphs")

#custom select chart
x_sel = st.selectbox(label="Select the X Value",
options=("income", "education", "parent", "married", "female", "age"))

y_sel = st.selectbox(label="Select the Y Value",
options=("income", "education", "parent", "married", "female", "age"))

if st.checkbox("Show Your Chart"):
     alt_plot = alt.Chart(ss).mark_bar().encode(x=x_sel,
                                                y=f"mean({y_sel})").interactive()
     alt_plot



######################################################
#Now, allow for users to input their parameters to do prediction
st.markdown("***")
st.markdown("***")

st.markdown("## Predict whether a User is on LinkedIn:")

#income
income_answer = st.selectbox(label="Select the Income Range",
options=("0 to $9,999","$10,000 to $19,999", "$20,000 to $29,999", "$30,000 to $39,999", "$40,000 to $49,999", "$50,000 to $74,999", 
         "$75,000 to $99,999",
        "$100,000 to $149,999",
        "$150,000+"))
if income_answer == "0 to $9,999":
    income_answer = 1
elif income_answer == "$10,000 to $19,999":
    income_answer = 2
elif income_answer == "$20,000 to $29,999":
    income_answer = 3
elif income_answer == "$30,000 to $39,999":
    income_answer = 4
elif income_answer == "$40,000 to $49,999":
    income_answer = 5
elif income_answer == "$50,000 to $74,999":
    income_answer = 6
elif income_answer == "$75,000 to $99,999":
    income_answer = 7
elif income_answer == "$100,000 to $149,999":
    income_answer = 8
elif income_answer == "$150,000+":
    income_answer = 9


#education

education_answer = st.selectbox(label="Select the Education Level",
options=("Less than high school (Grades 1-8 or no formal schooling)",
         "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)", 
         "High school graduate (Grade 12 with diploma or GED certificate)", 
         "Some college, no degree (includes some community college)", 
         "Two-year associate degree from a college or university", 
         "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)", 
         "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
        "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"))
if education_answer == "Less than high school (Grades 1-8 or no formal schooling)":
    education_answer = 1
elif education_answer == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education_answer = 2
elif education_answer == "High school graduate (Grade 12 with diploma or GED certificate)":
    education_answer = 3
elif education_answer == "Some college, no degree (includes some community college)":
    education_answer = 4
elif education_answer == "Two-year associate degree from a college or university":
    education_answer = 5
elif education_answer == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    education_answer = 6
elif education_answer == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education_answer = 7
elif education_answer == "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
    education_answer = 8



#parent?
parent_answer = st.selectbox(label="Are they a parent?", options=("Yes", "No"))

if parent_answer == "Yes":
    parent_answer =1 
else:
    parent_answer = 0

#married?
mar_answer = st.selectbox(label="Are they married?", options=("Yes", "No"))

if mar_answer == "Yes":
    mar_answer =1 
else:
    mar_answer = 0

#female
fem_answer = st.selectbox(label="Are they female?", options=("Yes", "No"))

if fem_answer == "Yes":
    fem_answer =1 
else:
    fem_answer = 0

#Age
age_answer = st.slider("Age", min_value = 1, max_value =100)

###############################################################
#Write the function to take these inputs and run a prediction

def predict(income, education, parent, married, female, age):
    pred_person = pd.DataFrame([[income, education, parent, married, female, age]],
                               columns=x.columns)
    probs = model.predict_proba(pred_person)
    predicted_class = model.predict(pred_person)
    return(probs, predicted_class)




#Submit and run the prediction
if st.button("Click to Predict"):
    probs, predicted_class = predict(income_answer, education_answer, parent_answer, mar_answer,fem_answer, age_answer)
    st.write("The probability of this person being a LinkedIn User is")
    st.write(f"Percentage:", round(probs[0][1]*100,2), "%")
    st.write("The predicted LinkedIn Status Is:")
    if predicted_class == 0:
        st.markdown("#### Not a LinkedIn User")
    else:
        st.markdown("#### LinkedIn User")

##################################################################
#Interactive chart code to use the inputs defined above
st.markdown("***")
st.markdown("***")

st.markdown("#### The chart below shows the percentage change in the predicted likelihood of being a LinkedIn user as you adjust different user characteristics.")
# The person described by the app's inputs (use your own input variable names here)
person = {"income": income_answer, "education": education_answer, "parent": parent_answer,
          "married": mar_answer, "female": fem_answer, "age": age_answer}

# Pick which predictor goes on the x-axis
var = st.selectbox("View User Characteristics",
                   ["income", "education", "parent", "married", "female", "age"])

# Possible values for each predictor
ranges = {"income": range(1, 10), "education": range(1, 9), "parent": [0, 1],
          "married": [0, 1], "female": [0, 1], "age": range(18, 98)}

# Copy the person once per value, changing only the chosen predictor
people = pd.DataFrame([person] * len(ranges[var]))
people[var] = list(ranges[var])

# predicted chance for each value
people["chance"] = model.predict_proba(
    people[["income", "education", "parent", "married", "female", "age"]])[:, 1] * 100

# Plot
people["label"] = people["chance"].round(1).astype(str) + "%"

if var == "age":
    line = alt.Chart(people).mark_line(point=True).encode(
        x=var,
        y="chance")
    st.altair_chart(line, use_container_width=True)
else:
    line = alt.Chart(people).mark_line(point=True).encode(
        x=alt.X(var, axis=alt.Axis(tickMinStep=1)),
        y="chance")
    labels = alt.Chart(people).mark_text(dy=-10, color="#FFFFFF").encode(
        x=var,
        y="chance",
        text="label")
    st.altair_chart(line + labels, use_container_width=True)

#Heatmap chart
rows = []

for inc in range(1, 10):
    for fem in [0, 1]:
        rows.append([inc, education_answer, parent_answer, mar_answer, fem, age_answer])

grid = pd.DataFrame(rows, columns=["income", "education", "parent", "married", "female", "age"])
grid["probability"] = model.predict_proba(grid)[:, 1] * 100

heat = alt.Chart(grid).mark_rect().encode(
    x="income:O",
    y="female:O",
    color="probability:Q",
    tooltip=["income", "female", "probability"]
)
st.altair_chart(heat)