\# 🏨 ANN Hotel Booking Cancellation Prediction



An \*\*Artificial Neural Network (ANN)-based Machine Learning web application\*\* that predicts whether a hotel booking is likely to be cancelled based on booking, guest, room, and reservation-related information.



The project uses a trained ANN model with \*\*248 input features\*\* and provides an interactive prediction interface built with \*\*Streamlit\*\*.



\---



\## 📌 Project Overview



Hotel booking cancellations can affect hotel revenue, room availability, staff planning, and overall business operations.



This project uses historical hotel booking data to train an \*\*Artificial Neural Network\*\* that learns patterns associated with booking cancellations.



The trained model is integrated into a Streamlit application where users can enter booking details and receive:



\* 📊 Cancellation probability

\* 🎯 Cancellation prediction

\* ✅ Real-time prediction through a web interface



\### Example Prediction



For one sample booking, the application produced:



\*\*Cancellation Probability:\*\* 77.35%



\*\*Prediction:\*\* CANCELLED



> The prediction probability represents the model's estimated likelihood that the entered booking will be cancelled.



\---



\## ✨ Features



\* 🤖 Artificial Neural Network based prediction

\* 🏨 Hotel booking cancellation prediction

\* 📊 248 model input features

\* 🎯 Cancellation probability output

\* 🖥️ Interactive Streamlit web interface

\* 📋 Booking and guest information input

\* 🛏️ Room and reservation details

\* ⚡ Fast prediction using a pre-trained model

\* 💾 Saved model and preprocessing files

\* 📓 Complete model development notebook



\---



\## 🛠️ Technologies Used



| Technology         | Purpose                               |

| ------------------ | ------------------------------------- |

| Python             | Programming language                  |

| Pandas             | Data processing                       |

| NumPy              | Numerical computation                 |

| Scikit-learn       | Data preprocessing                    |

| TensorFlow / Keras | Artificial Neural Network             |

| Joblib             | Saving/loading preprocessing objects  |

| Streamlit          | Web application                       |

| Matplotlib         | Data visualization                    |

| Seaborn            | Data visualization                    |

| Jupyter Notebook   | Model development and experimentation |

| Git \& GitHub       | Version control                       |



\---



\## 🧠 Machine Learning Model



The project uses an \*\*Artificial Neural Network (ANN)\*\* implemented using TensorFlow/Keras.



The model was trained to learn the relationship between hotel booking characteristics and the cancellation target.



\### Model Architecture



```text

Input Layer

&#x20;   ↓

Dense Layer - 128 neurons

&#x20;   ↓

Dropout

&#x20;   ↓

Dense Layer - 64 neurons

&#x20;   ↓

Dropout

&#x20;   ↓

Dense Layer - 32 neurons

&#x20;   ↓

Output Layer - 1 neuron

&#x20;   ↓

Cancellation Probability

```



The trained model contains approximately \*\*42,241 parameters\*\*.



\---



\## 📊 Dataset



The project is based on the \*\*Hotel Booking Demand\*\* dataset.



The original dataset contains approximately:



```text

Records: 119,390

Columns: 32

```



During preprocessing, duplicate and unsuitable records were handled before training.



The processed dataset was transformed into numerical features suitable for neural-network training.



After feature encoding, the model uses:



```text

248 Features

```



\### Dataset Processing



The major preprocessing steps include:



1\. Loading the hotel booking dataset

2\. Checking missing values

3\. Handling duplicate records

4\. Removing/handling unsuitable data

5\. Encoding categorical variables

6\. Preparing numerical features

7\. Splitting the dataset into training and testing sets

8\. Scaling the input features

9\. Training the ANN model



\---



\## 🔄 Machine Learning Workflow



```text

Hotel Booking Dataset

&#x20;       │

&#x20;       ▼

Data Cleaning

&#x20;       │

&#x20;       ▼

Duplicate Handling

&#x20;       │

&#x20;       ▼

Categorical Encoding

&#x20;       │

&#x20;       ▼

Feature Preparation

&#x20;       │

&#x20;       ▼

Feature Scaling

&#x20;       │

&#x20;       ▼

Train / Test Split

&#x20;       │

&#x20;       ▼

ANN Model Training

&#x20;       │

&#x20;       ▼

Model Evaluation

&#x20;       │

&#x20;       ▼

Save Trained Model

&#x20;       │

&#x20;       ▼

Streamlit Application

&#x20;       │

&#x20;       ▼

User Booking Details

&#x20;       │

&#x20;       ▼

Preprocessing

&#x20;       │

&#x20;       ▼

ANN Prediction

&#x20;       │

&#x20;       ▼

Cancellation Probability

&#x20;       │

&#x20;       ▼

CANCELLED / NOT CANCELLED

```



\---



\## 📁 Project Structure



```text

ANN-Hotel-Booking-Prediction/

│

├── app.py

│

├── hotel\_booking.ipynb

│

├── hotel\_booking\_ann\_model.keras

│

├── hotel\_booking\_scaler.pkl

│

├── hotel\_booking\_features.pkl

│

├── hotel\_booking\_feature\_columns.pkl

│

├── requirements.txt

│

├── .gitignore

│

└── README.md

```



\---



\## 📄 File Description



\### `app.py`



The main Streamlit application.



It:



\* Loads the trained ANN model

\* Loads preprocessing objects

\* Accepts booking information from the user

\* Prepares the input data

\* Performs prediction

\* Displays cancellation probability

\* Displays the final prediction



\### `hotel\_booking.ipynb`



Jupyter Notebook containing the machine-learning workflow, including:



\* Data loading

\* Data exploration

\* Data preprocessing

\* Feature engineering

\* Encoding

\* Scaling

\* ANN model creation

\* Model training

\* Evaluation

\* Model saving



\### `hotel\_booking\_ann\_model.keras`



The trained TensorFlow/Keras ANN model used by the Streamlit application.



\### `hotel\_booking\_scaler.pkl`



Saved feature-scaling object used to transform user input in the same way as the training data.



\### `hotel\_booking\_features.pkl`



Saved feature information used by the prediction system.



\### `hotel\_booking\_feature\_columns.pkl`



Saved feature-column information used to maintain consistency between training data and application input.



\### `requirements.txt`



Contains the Python dependencies required to run the project.



\---



\# 💻 Installation



\## 1. Clone the Repository



Open Command Prompt or PowerShell:



```bash

git clone https://github.com/ssdp9598-source/ANN-Hotel-Booking-Prediction.git

```



Move into the project directory:



```bash

cd ANN-Hotel-Booking-Prediction

```



\---



\## 2. Create a Virtual Environment



```bash

python -m venv .venv

```



\### Windows



Activate the environment:



```bash

.venv\\Scripts\\activate

```



\### Linux / macOS



```bash

source .venv/bin/activate

```



\---



\## 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\## 4. Run the Streamlit Application



```bash

streamlit run app.py

```



The application will normally open at:



```text

http://localhost:8501

```



\---



\# 🖥️ Using the Application



\### Step 1 — Start the application



Run:



```bash

streamlit run app.py

```



\### Step 2 — Enter Booking Details



Enter the required information in the Streamlit interface.



The application contains sections for booking-related information such as:



\* 📝 Booking Details

\* 🛏️ Room \& Guest Details

\* 📊 Additional Details



\### Step 3 — Generate Prediction



Submit the information to generate a prediction.



The system calculates a cancellation probability using the trained ANN model.



\### Step 4 — View Result



The application displays:



```text

Cancellation Probability

&#x20;       ↓

&#x20;     XX.XX%



Prediction

&#x20;       ↓

CANCELLED / NOT CANCELLED

```



\---



\# 🎯 Sample Result



Example output from the application:



```text

Cancellation Probability

77.35%



Prediction

CANCELLED



Booking is likely to be CANCELLED

```



\---



\# 📈 Model Performance



The project includes model training and evaluation inside:



```text

hotel\_booking.ipynb

```



The notebook should be used as the source of truth for detailed evaluation metrics such as:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Loss

\* Validation performance



> \*\*Note:\*\* Evaluation metrics are intentionally not hard-coded in this README unless they are verified directly from the training notebook.



\---



\# 🔐 Important Note About Prediction Probability



The displayed percentage is the model's predicted probability for cancellation.



For example:



```text

77.35%

```



means the model estimated a relatively high likelihood of cancellation for that particular input.



The prediction should be treated as a \*\*machine-learning estimate\*\*, not a guarantee that the booking will actually be cancelled.



\---



\# 🌐 Streamlit Application



The application is designed as an interactive web interface using Streamlit.



\### Main Application Flow



```text

User

&#x20;│

&#x20;▼

Enter Booking Information

&#x20;│

&#x20;▼

Streamlit Application

&#x20;│

&#x20;▼

Feature Preparation

&#x20;│

&#x20;▼

Saved Scaler

&#x20;│

&#x20;▼

ANN Model

&#x20;│

&#x20;▼

Prediction Probability

&#x20;│

&#x20;▼

Final Result

```



\---



\# 📚 Learning Objectives



This project demonstrates practical knowledge of:



\* Python programming

\* Data preprocessing

\* Exploratory data analysis

\* Categorical feature encoding

\* Feature scaling

\* Machine learning

\* Artificial Neural Networks

\* TensorFlow/Keras

\* Model serialization

\* Streamlit application development

\* Git and GitHub

\* Machine-learning model integration



\---



\# 🔮 Future Improvements



Possible future improvements include:



\* 📊 Add more model evaluation visualizations

\* 📈 Add confusion matrix and classification report

\* 📉 Add ROC-AUC evaluation

\* 🎨 Improve Streamlit UI/UX

\* 📊 Add prediction history

\* 💾 Store prediction results

\* 📥 Add CSV batch prediction

\* 📊 Add interactive charts

\* 🔐 Add user authentication

\* ☁️ Deploy the application online

\* 🔄 Add automated model retraining

\* 🧪 Add more comprehensive model testing



\---



\# ⚠️ Limitations



\* Predictions depend on the quality of the training dataset.

\* The model may not perform equally well on every type of hotel booking.

\* A predicted cancellation probability is not a guarantee.

\* The model should be evaluated with appropriate validation and test metrics before being used for real business decisions.



\---



\# 👨‍💻 Author



\*\*Dipendra Yadav\*\*



MCA Student | Python | Machine Learning | Full Stack Development



\---



\# ⭐ Project



If you find this project useful for learning about Artificial Neural Networks and hotel booking prediction, consider giving the repository a ⭐ on GitHub.



\---



\## 📜 License



This project is intended for \*\*educational and academic purposes\*\*.



You may modify and use the project for learning and educational work while respecting the licenses of the datasets and third-party libraries used.



