import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to visualize an image based on the label
def visualize_digit(data, label):
    digit_data = data[data['label'] == label].iloc[0, 1:].values.astype(int)
    digit_image = digit_data.reshape(28, 28)

    plt.imshow(digit_image, interpolation=None, cmap="gray")
    plt.title(f"Visualization of digit {label}")
    plt.show()

# Function to recognize a given image
def recognize_digit(model, image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = 255 - np.array(img).flatten()  # Flatten and invert the pixel values

    # Make prediction using the SVM model
    prediction = model.predict([img_array])[0]
    print(f"The recognized digit is: {prediction}")
    return prediction

# Function to calculate recognition accuracy rate
def calculate_accuracy(model, X_test, Y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    return accuracy

# Function to display the confusion matrix
def display_confusion_matrix(model, X_test, Y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(Y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Get data from CSV 
dataframe = pd.read_csv(r"C:\Users\murar\Downloads\Handwritten-Digits-Recognition-using-SVM--master (1)\Handwritten-Digits-Recognition-using-SVM--master\csv\dataset6labels.csv")
dataframe = shuffle(dataframe).reset_index(drop=True)

# Separate Labels and Features
X = dataframe.drop(['label'], axis=1)
Y = dataframe['label']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = svm.SVC(kernel="linear")
print("Fitting. This might take some time...")
model.fit(X_train, Y_train)

# Save the model
joblib.dump(model, r"C:\Users\murar\Downloads\Handwritten-Digits-Recognition-using-SVM--master (1)\Handwritten-Digits-Recognition-using-SVM--master\model\svm_6label_linear")

# Add a new instance to the dataset
new_image_path = r"C:\Users\murar\OneDrive\Desktop\six.png"
new_label = 6
new_image = Image.open(r"C:\Users\murar\OneDrive\Desktop\six.png").convert('L')
new_image = new_image.resize((28, 28))
new_array = 255 - np.array(new_image).flatten()

# Add the new instance to the dataset
new_row = pd.DataFrame([np.append([new_label], new_array)], columns=dataframe.columns)
dataframe = dataframe._append(new_row, ignore_index=True)

# Save the updated dataset to CSV
dataframe.to_csv(r"C:\Users\murar\Downloads\Handwritten-Digits-Recognition-using-SVM--master (1)\Handwritten-Digits-Recognition-using-SVM--master\csv\dataset6labels.csv", index=False)

# Visualize the new digit
visualize_digit(dataframe, new_label)

# Recognize the new digit
recognized_digit = recognize_digit(model, new_image_path)

# Calculate recognition accuracy rate
accuracy = calculate_accuracy(model, X_test, Y_test)
print(f"Recognition accuracy rate on the test set: {accuracy}")

# Display the confusion matrix
display_confusion_matrix(model, X_test, Y_test)
