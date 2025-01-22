import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify
import joblib
import os

# Reading the dataset
data = pd.read_csv(r'urlset.csv', encoding='ISO-8859-1', on_bad_lines='skip')
app = Flask(__name__)

# Data Cleaning
data = data.drop_duplicates().dropna()

# Selecting the 7 numeric columns for scaling (adjust as needed)
numeric_columns = ['card_rem', 'ratio_Rrem', 'ratio_Arem', 
                   'jaccard_RR', 'jaccard_RA', 'jaccard_AR', 
                   'jaccard_AA']

# Outlier Handling using IQR
for col in numeric_columns:
    Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower=lower, upper=upper)

# Encoding categorical features
label_encoder = LabelEncoder()
object_cols = data.select_dtypes(include='object').columns.tolist()

for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Splitting features and labels (using 7 features)
X = data[['card_rem', 'ratio_Rrem', 'ratio_Arem', 'jaccard_RR', 'jaccard_RA', 'jaccard_AR', 'jaccard_AA']].values  # 7 features
y = data['label'].values

# Normalizing features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Creating DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Bahdanau Attention
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, attention_size)
        self.W2 = nn.Linear(hidden_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, hidden_state, encoder_outputs):
        hidden_state = hidden_state.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_state)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context_vector, attention_weights

# GRU Model with Attention
class GRUModelWithAttention(nn.Module):
    def __init__(self, input_dim, gru_units, attention_units, dropout_rate):
        super(GRUModelWithAttention, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_units, num_layers=2,
                          batch_first=True, dropout=dropout_rate)
        self.attention = BahdanauAttention(gru_units, attention_units)
        self.fc1 = nn.Linear(gru_units, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        hidden_state = gru_out[:, -1, :]
        context_vector, _ = self.attention(hidden_state, gru_out)
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Model Initialization
input_dim = X_train.shape[1]  # This should be 7 based on the selected features
gru_units = 64
attention_units = 32
dropout_rate = 0.2
learning_rate = 0.001
epochs = 20

model = GRUModelWithAttention(input_dim, gru_units, attention_units, dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if the model is already trained and saved
model_file = "gru_attention_model.pth"
scaler_file = "scaler.pkl"

# Training Loop (Only runs once if the model is not trained yet)
def train_model():
    if not os.path.exists(model_file):  # Train only if model is not already saved
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        # Save the model and scaler after training
        torch.save(model.state_dict(), model_file)
        joblib.dump(scaler, scaler_file)
        print("Model and Scaler saved!")
    else:
        print("Model already trained and saved.")

# Load the model and scaler (Done once when the app starts)
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    print("Model loaded!")
else:
    train_model()  # Train the model if it's not loaded

scaler = joblib.load(scaler_file)  # Load the scaler
model.eval()  # Set the model to evaluation mode

# Prediction function
def predict(input_data):
    # Convert the input JSON into a numpy array
    input_features = np.array([[input_data['card_rem'], input_data['ratio_Rrem'], input_data['ratio_Arem'],
                                input_data['jaccard_RR'], input_data['jaccard_RA'], input_data['jaccard_AR'], 
                                input_data['jaccard_AA']]])

    # Scale the input using the saved scaler
    input_features_scaled = scaler.transform(input_features)

    # Convert to tensor
    input_tensor = torch.tensor(input_features_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output.squeeze() > 0.5).long()  # Binary classification (0 or 1)

    return prediction.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get input data from form
        input_data = {
            'card_rem': float(request.form['card_rem']),
            'ratio_Rrem': float(request.form['ratio_Rrem']),
            'ratio_Arem': float(request.form['ratio_Arem']),
            'jaccard_RR': float(request.form['jaccard_RR']),
            'jaccard_RA': float(request.form['jaccard_RA']),
            'jaccard_AR': float(request.form['jaccard_AR']),
            'jaccard_AA': float(request.form['jaccard_AA'])
        }

        # Get prediction
        prediction = predict(input_data)

    return render_template('index.html', prediction=prediction)

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Get the JSON data from the POST request
        input_data = request.get_json()

        # Ensure all necessary keys are provided in the input data
        required_keys = ['card_rem', 'ratio_Rrem', 'ratio_Arem', 'jaccard_RR', 'jaccard_RA', 'jaccard_AR', 'jaccard_AA']
        if not all(key in input_data for key in required_keys):
            return jsonify({'error': 'Missing input data'}), 400

        # Get the prediction
        prediction = predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
