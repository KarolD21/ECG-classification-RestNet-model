import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sn
from data_input_v2 import df_ptbxl_statements
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Readind data from fast .gzip file
name = '100'

df_ecg_training = pd.read_parquet(f'gzip_data/df_ecg_training{name}.gzip', engine='fastparquet')
df_ecg_validation = pd.read_parquet(f'gzip_data/df_ecg_validation{name}.gzip', engine='fastparquet')
df_ecg_testing = pd.read_parquet(f'gzip_data/df_ecg_testing{name}.gzip', engine='fastparquet')

# RESIDUAL BLOCK

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
            )
        self.shortcut = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
                )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# In above lines one residual block have been defined. It contains
# two layers of size 3x3 and the 'shortcut' where the additional convolution and normalization
# may occur in case of inconsistency of input and output sizes between successive layers
    
# MODEL

class ResNet34Model(nn.Module):
    def __init__(self, block, layers, input_dim, num_classes):
        super(ResNet34Model, self).__init__()
        self.in_channels = 64 # number o input channels/filters/feature maps
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.maxpooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layerConv2 = self.make_layer(block, 64, layers[0], stride=1)
        # block (type of residual block), 64 (output channels), layers[0] (number of blocks), and a stride of 1
        self.layerConv3 = self.make_layer(block, 128, layers[1], stride=2)
        self.layerConv4 = self.make_layer(block, 256, layers[2], stride=2)
        self.layerConv5 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        # self.avg_pooling = nn.AvgPool1d(7, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(512, num_classes) # fully connected layer - all neurons connected

    def make_layer(self, block, out_channels, num_blocks, stride):
        # block: The type of residual block to be used (in this case, ResidualBlock or Bottleneck)
        # num_blocks: The number of residual blocks to create within the layer
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # updates the in_channels attribute to out_channels, as subsequent residual blocks in the
        # layer will have out_channels as their input dimension
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpooling(out)
        out = self.layerConv2(out) # 64 filters (identity and residual connections inside)
        # out = self.dropout(out) # Dropout after first few layers to prevent overfitting early
        out = self.layerConv3(out)
        # out = self.maxpooling(out)  # Additional pooling to downsample
        out = self.layerConv4(out)
        # out = self.maxpooling(out)  # Further downsample with pooling
        out = self.layerConv5(out)
        out = self.avg_pooling(out)
        out = self.dropout(out) # Dropout after whole cycle prevent overfitting
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)

        return out

train_losses_global = []
train_accuracy_global =[]
val_losses_global = []
val_accuracy_global = []

# TRAINING FUNCTION

def trainging(model, training_loader, validation_loader, criterion, optimizer, scheduler, epoch, num_epoch, task, patience):
    
    model.train()
    train_loss = 0.0 # accumulate the training loss over all batches in the epoch

    global best_loss, counter, early_stop, train_losses, train_accuracy, val_losses, val_accuracy, min_delta

    # Early stopping parameters
    if epoch+1 == 1:
        best_loss = float('inf')
        counter = 0
        min_delta = 0.1 # Minimum change in loss to qualify as an improvement
        early_stop = False
        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []
        # Printing device name
        print('Using device:', device)
        if device.type == 'cuda':
            print('Device name:', torch.cuda.get_device_name())
        print('Task category: ', str(task))
        print()

    correct_train = 0
    total_train = 0

    for train_input, train_target in training_loader:
        train_input = train_input.to(device)
        train_target = train_target.to(device)
        train_output = model(train_input)

        loss = criterion(train_output, train_target) # loss between predictions and actual target (MSE)
        optimizer.zero_grad() # zeroing the gradient in smoothing functions
        train_loss += loss.item() * train_input.size(0)
                                  # '.item()' extracts the scalar value of the loss tensor and 
                                  # adds the current batch's loss to the total training loss for the epoch
                                  # '* inputs.size(0)' scaling the loss in case when the last batches are
                                  # smaller then default
        _, predicted_train = torch.max(train_output, 1)
        total_train += train_target.size(0)
        correct_train += (predicted_train == train_target).sum().item()

        loss.backward()
        optimizer.step() # gradient weights update

    avg_train_loss = train_loss / len(training_loader) # average loss for every batches in single epoch
    avg_train_accuracy = 100 * correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracy.append(avg_train_accuracy)

    print(f'Epoch [{epoch+1:03}/{num_epoch:03}] \nTrain loss: {avg_train_loss:.3f} \nTrain accuaracy: {avg_train_accuracy:.2f}%')

    model.eval()
    correct_val = 0
    total_val = 0
    all_predicted = []
    all_targets = []
    val_loss = 0.0

    with torch.no_grad():
        for val_input, val_target in validation_loader:
            val_input = val_input.to(device)
            val_target = val_target.to(device)
            val_output = model(val_input)

            loss = criterion(val_output, val_target)
            val_loss += loss.item() * val_input.size(0)
            _, predicted_val = torch.max(val_output, 1) # explained in writing notes
            total_val += val_target.size(0) # count the number of outputs for single batch
            correct_val += (predicted_val == val_target).sum().item()

            all_predicted.extend(predicted_val.cpu().numpy())
            all_targets.extend(val_target.cpu().numpy())
    
    avg_val_loss = val_loss / len(validation_loader)
    avg_val_accuracy = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracy.append(avg_val_accuracy)
    print(f'Validation loss: {avg_val_loss:.3f} \nValidation Accuracy: {avg_val_accuracy:.2f}%')

    # Multi-class classification, calculate precision and recall for each class
    all_predicted = np.array(all_predicted)
    all_targets = np.array(all_targets)

    precision = precision_score(all_targets, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predicted, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predicted, average='weighted', zero_division=0)

    print(f"Precision per class: {precision:.4f}")
    print(f"Recall per class: {recall:.4f}")
    print(f"F1-Score per class: {f1:.4f}")

    # LR scheduler step
    scheduler.step(avg_train_loss)

    # Check for early stopping
    if avg_val_loss < best_loss - min_delta:
        best_loss = avg_val_loss
        counter = 0  # Reset counter if there is an improvement
    else:
        counter += 1  # Increment counter if no improvement
        if counter >= patience:
            print(f"Early stopping at epoch [{epoch+1:03}/{num_epoch:03}]")
            early_stop = True

    # Collection of the results to global list
    if epoch + 1 == num_epoch or early_stop:
        train_losses_global.append(train_losses)
        train_accuracy_global.append(train_accuracy)
        val_losses_global.append(val_losses)
        val_accuracy_global.append(val_accuracy)
        # Detailed classification report
        print(classification_report(all_targets, all_predicted, zero_division=0))
        if device.type == 'cuda':
            print()
            print('Last memory usage:')
            print('Allocated:', round(torch.cuda.memory_allocated()/1024**3,4), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved()/1024**3,4), 'GB')
            print()
            print('------------------------------------------------------------')
            print()

    return early_stop

# HYPERPARAMETERS   

batch_size_diag = 32
batch_size_rhythm = 32
batch_size_form = 32

num_epoch_diag = 70
num_epoch_rhythm = 70
num_epoch_form = 100

lr = 1e-3
lr2 = 1e-5
input_dim = 14 # number of features / number of neurons in input layer

# number of neurons in output layer
num_classes_diag = 44 # min: 44
num_classes_rhythm = 13 # min: 13
num_classes_form = 20 # min: 20

weight_decay = 1e-4
momentum = 0.7
momentum2 = 0.9

model_diag_task = ResNet34Model(ResidualBlock, [3, 4, 6, 3], input_dim, num_classes_diag)
model_rhythm_task = ResNet34Model(ResidualBlock, [3, 4, 6, 3], input_dim, num_classes_rhythm)
model_form_task = ResNet34Model(ResidualBlock, [3, 4, 6, 3], input_dim, num_classes_form)

criterion = nn.CrossEntropyLoss()

optimizer_diag_task = torch.optim.SGD(model_diag_task.parameters(), lr=lr, momentum=momentum)
optimizer_rhythm_task = torch.optim.SGD(model_rhythm_task.parameters(), lr=lr, momentum=momentum)
optimizer_form_task = torch.optim.SGD(model_form_task.parameters(), lr=lr2, momentum=momentum2)

scheduler_diag_task = ReduceLROnPlateau(optimizer_diag_task, mode='min', factor=0.5, patience=2)
scheduler_rhythm_task = ReduceLROnPlateau(optimizer_rhythm_task, mode='min', factor=0.5, patience=2)
scheduler_form_task = ReduceLROnPlateau(optimizer_form_task, mode='min', factor=0.5, patience=2)

# DATALOADER

class ECG_Dataset(Dataset):
    def __init__(self, df, category): # initialising (assigning values) to the data pieces of the class when the class object is created
        super().__init__()

        self.data = df
        self.layers = self.data.layer.unique()
        self.category = category
        
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, index):
        layer_idx = self.layers[index]

        signal_data = self.data[(self.data.layer == layer_idx) & (self.data.row < 14)]
        label_data = self.data[(self.data.layer == layer_idx) & (self.data.row >=14)]

        ecg_signal = torch.tensor(signal_data.iloc[:, 2:].values, dtype=torch.float32)
        
        if self.category == 'diagnostic':
            target_task_diagnostic = torch.tensor(label_data.iloc[2, 2], dtype=torch.int64) # diagnostic
            return ecg_signal, target_task_diagnostic
        elif self.category == 'rhythm':
            target_task_rhythm = torch.tensor(label_data.iloc[1, 2], dtype=torch.int64) # rhythm
            return ecg_signal, target_task_rhythm
        else:
            target_task_form = torch.tensor(label_data.iloc[0, 2], dtype=torch.int64) # form
            return ecg_signal, target_task_form
    
train_dataset_diagnostic = ECG_Dataset(df_ecg_training, category='diagnostic')
test_dataset_diagnostic = ECG_Dataset(df_ecg_testing, category='diagnostic')
val_dataset_diagnostic = ECG_Dataset(df_ecg_validation, category='diagnostic')

train_dataset_rhythm = ECG_Dataset(df_ecg_training, category='rhythm')
test_dataset_rhythm = ECG_Dataset(df_ecg_testing, category='rhythm')
val_dataset_rhythm = ECG_Dataset(df_ecg_validation, category='rhythm')

train_dataset_form = ECG_Dataset(df_ecg_training, category='form')
test_dataset_form = ECG_Dataset(df_ecg_testing, category='form')
val_dataset_form = ECG_Dataset(df_ecg_validation, category='form')

train_loader_diagnostic = DataLoader(dataset=train_dataset_diagnostic,
                          batch_size=batch_size_diag,
                          shuffle=True
                          )
test_loader_diagnostic = DataLoader(dataset=test_dataset_diagnostic,
                         batch_size=batch_size_diag,
                         shuffle=True
                         )
val_loader_diagnostic = DataLoader(dataset=val_dataset_diagnostic,
                        batch_size=batch_size_diag,
                        shuffle=True
                        )

train_loader_rhythm = DataLoader(dataset=train_dataset_rhythm,
                          batch_size=batch_size_rhythm,
                          shuffle=True
                          )
test_loader_rhythm = DataLoader(dataset=test_dataset_rhythm,
                         batch_size=batch_size_rhythm,
                         shuffle=True
                         )
val_loader_rhythm = DataLoader(dataset=val_dataset_rhythm,
                        batch_size=batch_size_rhythm,
                        shuffle=True
                        )

train_loader_form = DataLoader(dataset=train_dataset_form,
                          batch_size=batch_size_form,
                          shuffle=True
                          )
test_loader_form = DataLoader(dataset=test_dataset_form,
                         batch_size=batch_size_form,
                         shuffle=True
                         )
val_loader_form = DataLoader(dataset=val_dataset_form,
                        batch_size=batch_size_form,
                        shuffle=True
                        )

# TRAINING TASK DIAGNOSTIC

model_diag_task.to(device)
for epoch in range(num_epoch_diag):
    stop = trainging(model_diag_task, train_loader_diagnostic, val_loader_diagnostic, criterion, optimizer_diag_task, 
                     scheduler_diag_task, epoch, num_epoch_diag, patience=70, task='diagnostic')
    current_lr = scheduler_diag_task.get_last_lr()[0]
    print(f"Learning Rate: {current_lr:.5f}")
    print()
    if stop:
        break

# TRAINING TASK RHYTHM

model_rhythm_task.to(device)
for epoch in range(num_epoch_rhythm):
    stop = trainging(model_rhythm_task, train_loader_rhythm, val_loader_rhythm, criterion, optimizer_rhythm_task, 
                     scheduler_rhythm_task, epoch, num_epoch_rhythm, patience=70, task='rhythm')
    current_lr = scheduler_rhythm_task.get_last_lr()[0]
    print(f"Learning Rate: {current_lr:.5f}")
    print()
    if stop:
        break

# TRAINING TASK FORM

model_form_task.to(device)
for epoch in range(num_epoch_form):
    stop = trainging(model_form_task, train_loader_form, val_loader_form, criterion, optimizer_form_task, 
                     scheduler_form_task, epoch, num_epoch_form, patience=8, task='form')
    current_lr = scheduler_form_task.get_last_lr()[0]
    print(f"Learning Rate: {current_lr:.5f}")
    print()
    if stop:
        break

# Saving the model
torch.save(model_diag_task.state_dict(), f'models/ResNet34_model_diag_task_{name}.pth')
torch.save(model_rhythm_task.state_dict(), f'models/ResNet34_model_rhythm_task_{name}.pth')
torch.save(model_form_task.state_dict(), f'models/ResNet34_model_form_task_{name}.pth')

os.system('shutdown /s /t 1200')

# LOSS

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_losses_global[0])), train_losses_global[0], label='Task 1 training - diagnostic')
plt.plot(range(len(val_losses_global[0])), val_losses_global[0], label='Task 1 validation - diagnostic')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'Results/task_1_loss_{name}.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_losses_global[1])), train_losses_global[1], label='Task 2 training - rhythm')
plt.plot(range(len(val_losses_global[1])), val_losses_global[1], label='Task 2 validation - rhythm')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'Results/task_2_loss_{name}.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_losses_global[2])), train_losses_global[2], label='Task 3 training - form')
plt.plot(range(len(val_losses_global[2])), val_losses_global[2], label='Task 3 validation - form')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig(f'Results/task_3_loss_{name}.png')
plt.show()

# ACCURACY

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_accuracy_global[0])), train_accuracy_global[0], label='Task 1 training - diagnostic')
plt.plot(range(len(val_accuracy_global[0])), val_accuracy_global[0], label='Task 1 validation - diagnostic')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(f'Results/task_1_accuracy_{name}.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_accuracy_global[1])), train_accuracy_global[1], label='Task 2 training - rhythm')
plt.plot(range(len(val_accuracy_global[1])), val_accuracy_global[1], label='Task 2 validation - rhythm')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(f'Results/task_2_accuracy_{name}.png')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(range(len(train_accuracy_global[2])), train_accuracy_global[2], label='Task 3 training - form')
plt.plot(range(len(val_accuracy_global[2])), val_accuracy_global[2], label='Task 3 validation - form')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(f'Results/task_3_accuracy_{name}.png')
plt.show()

# TESTING

def testing(model, testing_loader, map_name):
    correct = 0
    total = 0
    cf_predicted = []
    cf_target = []

    with torch.no_grad():
        for input, target in testing_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1) # explained in writing notes
            total += target.size(0) # count the number of outputs for single batch
            correct += (predicted == target).sum().item()

            cf_predicted.extend(predicted.cpu().numpy())
            cf_target.extend(target.cpu().numpy())

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')
    print()

    data_map = np.load(f'maps/{map_name}_map{name}.npy',allow_pickle='TRUE').item()

    class_model = np.unique(np.concatenate((cf_target, cf_predicted)))
    classes = []
    for i in class_model:
        g = list(data_map.keys())[list(data_map.values()).index(i)]
        classes.append(df_ptbxl_statements.malfunction_name[int(g)])

    cf_matrix = confusion_matrix(cf_target, cf_predicted)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes]).round(2)
    plt.figure(figsize = (23,15))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'Results/cf_{map_name}_{name}.png')
    plt.show()

testing(model_diag_task, test_loader_diagnostic, 'diagnostic')

testing(model_rhythm_task, test_loader_rhythm, 'rhythm')

testing(model_form_task, test_loader_form, 'form')