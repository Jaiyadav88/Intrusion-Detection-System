from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Image
st.image('What-Is-the-Intrusion-Detection-System-IDS-scaled.jpg')

# Streamlit UI
st.title('Intrusion Detection System')
st.write("Upload a dataset, select an algorithm, and predict whether a packet is malicious or benign.")

# File upload
uploaded_file = st.file_uploader("Upload Dataset (CSV file)", type=['csv'])

# Select algorithm
algorithm = st.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Naive Bayes", "SVC", "KNN", "Compare All"])

# Initialize a dictionary to store accuracy scores
accuracy_scores = {}

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Plot histograms
    st.write("Feature Distributions:")
    cols = df.columns[:]
    fig, axes = plt.subplots((len(cols) + 1) // 2, 2, figsize=(10, 14))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[df['label'] == 0][col], label='Benign', color='Blue', density=True, alpha=0.7)
        axes[i].hist(df[df['label'] == 1][col], label='Attack', color='Red', density=True, alpha=0.7)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Probability')
        axes[i].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Prepare data
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Function to train and evaluate model
    def train_evaluate_model(model, model_name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'### Accuracy Score ({model_name}): {accuracy:.4f}')
        st.write(f'### Classification Report ({model_name}):')
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix ({model_name})')
        st.pyplot(fig)
        accuracy_scores[model_name] = accuracy

    # Select and apply algorithm
    if algorithm != 'Compare All':
        if algorithm == 'Random Forest':
            model = RandomForestClassifier()
        elif algorithm == 'Logistic Regression':
            model = LogisticRegression()
        elif algorithm == 'Naive Bayes':
            model = GaussianNB()
        elif algorithm == 'SVC':
            model = SVC()
        elif algorithm == 'KNN':
            model = KNeighborsClassifier()

        train_evaluate_model(model, algorithm)

    else:
        models = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'Naive Bayes': GaussianNB(),
            'SVC': SVC(),
            'KNN': KNeighborsClassifier()
        }
        
        for model_name, model in models.items():
            train_evaluate_model(model, model_name)

        if accuracy_scores:
            fig, ax = plt.subplots()
            ax.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
            ax.set_xlabel('Algorithms')
            ax.set_ylabel('Accuracy Score')
            ax.set_title('Accuracy Comparison of Algorithms')
            st.pyplot(fig)

    # User input for packet details
    st.sidebar.header('Enter Packet Details')
    user_input = {}
    for col in X.columns:
        user_input[col] = st.sidebar.number_input(f'Enter {col}', value=0.0)
    
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_scaled = scaler.transform(user_input_df)

    if algorithm == 'Random Forest':
        prediction = model.predict(user_input_scaled)
    elif algorithm == 'Logistic Regression':
        prediction = model.predict(user_input_scaled)
    elif algorithm == 'Naive Bayes':
        prediction = model.predict(user_input_scaled)
    elif algorithm == 'SVC':
        prediction = model.predict(user_input_scaled)
    elif algorithm == 'KNN':
        prediction = model.predict(user_input_scaled)
    else:
        st.sidebar.write("Please select an algorithm to make predictions.")

    if 'prediction' in locals():
        if prediction[0] == 1:
            st.sidebar.write("### The packet is predicted to be: **Malicious**")
        else:
            st.sidebar.write("### The packet is predicted to be: **Benign**")
