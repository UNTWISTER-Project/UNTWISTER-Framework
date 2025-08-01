import torch
from sklearn.metrics import f1_score

def validate_lstm(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            val_loss += loss.item()

            # Collect predictions and targets
            preds = (outputs >= 0.5).float()  # Threshold at 0.5
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_f1 = f1_score(all_targets, all_preds)
    return val_loss / len(val_loader), val_f1


def train_lstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 3  # Stop training if val loss doesn't improve for 3 epochs

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()  # Remove extra dimensions
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss, val_f1 = validate_lstm(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_lstm_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break



def train_mlp(model, dataloader, loss_fn, optimizer, epochs=10, validation_data=None):
    model.train()  # Set the model to training mode
    
    if validation_data:
        val_inputs, val_targets = validation_data
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Training phase
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = loss_fn(outputs, targets.long())  # Ensure targets are long for CrossEntropy
            
            # Backward pass
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate
            optimizer.step()  # Update the model weights
            
            running_loss += loss.item()
        
        # Average training loss for the epoch
        avg_train_loss = running_loss / len(dataloader)
        
        # Validation phase
        if validation_data:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No gradient calculations
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_targets.long())
                
                # Calculate validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = (predicted == val_targets).float().mean().item()
            
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, '
                  f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')