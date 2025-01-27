import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer-based regression model
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=512):
        super(TransformerRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Embedding layer (for the input sequence of cosmological parameters)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward
            ), num_layers=num_layers
        )
        
        # Final linear layer to map the transformer output to the desired output
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # shape (batch_size, 1, d_model)
        x = x.permute(1, 0, 2)  # shape (1, batch_size, d_model) for transformer input
        
        # Pass through the transformer
        x = self.transformer(x)
        
        # Take the output of the first (and only) token
        x = x[0, :, :]
        
        # Output layer
        x = self.fc_out(x)
        return x


from sklearn.preprocessing import StandardScaler
from acm.data.io_tools import *

statistic = 'tpcf'
select_filters = {}
slice_filters = {}

X_train, y_train, coords = read_lhc(statistics=[statistic],
                                select_filters=select_filters,
                                slice_filters=slice_filters)
n_input = X_train.shape[1]
n_output = y_train.shape[1]
print(f'Loaded LHC with shape: {X_train.shape}, {y_train.shape}')

# Assuming X_train is your cosmological parameters (n_obs, n_input)
# and y_train is your galaxy correlation function (n_obs, n_output)

# Example: Model instantiation for your problem
model = TransformerRegressor(input_dim=n_input, output_dim=n_output, d_model=128, nhead=4, num_layers=4)
model.to(device)  # Move model to GPU

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Data loading
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# Convert to torch tensors and move to device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # Move the batch to the GPU
        inputs = inputs.to(device).unsqueeze(1)
        targets = targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation on the validation set
model.eval()  # Set to evaluation mode
with torch.no_grad():
    # Assuming X_val and y_val are your validation data
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    
    predictions = model(X_val_tensor)
    predictions = predictions.squeeze().cpu().numpy()  # Move predictions back to CPU for further processing

# You can now use predictions for further analysis