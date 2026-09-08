import nni 
from nni import get_next_parameter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_curve, confusion_matrix

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import joblib
from collections import Counter
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchinfo import summary

from utils.utils import *
import random
# import torch.nn.init as init

# Seed setup
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")


#read the data
def read_data(data_path):
    pd0=pd.read_csv(data_path+"/MLData2022.csv", sep=',')
    pd1=pd.read_csv(data_path+"/evalRuns.csv", sep=',')
    pd2=pd.read_csv(data_path+"/testRuns.csv", sep=',')

    pd1['classify_ae']=np.ones(pd1.shape[0])
    pd1['anomaly_new']=np.ones(pd1.shape[0])
    pd1['anomaly_new_adapt']=np.ones(pd1.shape[0])

    pd2['classify_ae']=np.zeros(pd2.shape[0])
    pd2['anomaly_new']=np.zeros(pd2.shape[0])
    pd2['anomaly_new_adapt']=np.zeros(pd2.shape[0])
    

    #getting the distances
    dist_q=np.array(pd2[['D_euc_vec_1', 'D_euc_vec_2', 'D_dtw_vec_1', 'D_dtw_vec_2']])
    dist_nq=np.array(pd1[['D_euc_vec_1', 'D_euc_vec_2', 'D_dtw_vec_1', 'D_dtw_vec_2']])
    dist_2022=np.array(pd0[['Var6_1', 'Var6_2', 'Var7_1', 'Var7_2']])
    print(dist_nq.shape)
    
    id=dist_nq[:, 0]!=0
    dist_nq=dist_nq[id, :]
    print(dist_nq.shape)
    

    # getting the ground truth
    gt_q=np.array(pd2['classify_ae'])
    gt_nq=np.array(pd1['classify_ae'])[id]
    print(gt_nq.shape)
    gt_2022=np.array(pd0[['Var8']])


    #split the data
    train =np.concatenate([dist_q, dist_nq, dist_2022[0:100000, :]], axis=0)
    test=dist_2022[100000:, :]

    gt_2022=gt_2022.reshape(gt_2022.shape[0],)
    print(f"{gt_q.shape}, {gt_nq.shape}, {gt_2022.shape}")
    gt_2022[gt_2022!=0]=1

    gt_train=np.concatenate([gt_q, gt_nq, gt_2022[0:100000]], axis=0)
    gt_test=gt_2022[100000:]
    # set positives to label 1, Swap 0s with 1s
    Ytrain = 1 - gt_train[:]
    Ytest = 1 - gt_test[:]  


    print(f" Train data description : {pd.Categorical(Ytrain).describe()}")
    # Y_train = tf.keras.utils.to_categorical(Ytrain, 2)
    # Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.long)  # Convert to tensor with dtype long
    # Y_train = torch.nn.functional.one_hot(Ytrain_tensor, num_classes=2).float()
    # print(Y_train)


    print(f" Test data description {pd.Categorical(Ytest).describe()}")
    # Y_test = tf.keras.utils.to_categorical(Ytest, 2)
    # Ytest_tensor = torch.tensor(Ytest, dtype=torch.long)  # Convert to tensor with dtype long
    # Y_test = torch.nn.functional.one_hot(Ytest_tensor, num_classes=2).float()
    # print(Y_test)
    return train, test, Ytrain, Ytest


# Define the MLP model based on NNI parameters
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_1_neurons, hidden_layer_2_neurons, hidden_layer_3_neurons, output_size):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_1_neurons))
        layers.append(nn.ReLU())
        
        if hidden_layer_2_neurons > 0:
            layers.append(nn.Linear(hidden_layer_1_neurons, hidden_layer_2_neurons))
            layers.append(nn.ReLU())
            previous_neurons = hidden_layer_2_neurons
        else:
            previous_neurons = hidden_layer_1_neurons

        if hidden_layer_3_neurons > 0:
            layers.append(nn.Linear(previous_neurons, hidden_layer_3_neurons))
            layers.append(nn.ReLU())
            previous_neurons = hidden_layer_3_neurons

        # Output layer
        layers.append(nn.Linear(previous_neurons, output_size))

        self.model = nn.Sequential(*layers)

        self.model.apply(self._initialize_weights)
        
    def _initialize_weights(self, layer):
            """Initialize weights and biases for Linear layers."""
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    train_losses = []
    grad_norms = []
    aucs=[]

    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0 
        epoch_grad_norms = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            loss.backward()

            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    total_grad_norm += grad_norm
            epoch_grad_norms.append(total_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()    
        
        if epoch % 10 == 0: 
            adjust_weight_decay(optimizer, epoch, decay_rate=0.9)
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        grad_norms.append(avg_grad_norm)

        print(f"************************************* Epoch {epoch+1}/{epochs} *************************************\n Loss: {epoch_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")

        auc, preds, labels = inference_on_test(model, val_loader, " validation data ")

            # Report accuracy to NNI
        nni.report_intermediate_result(auc)

    metric_auc_flops, flops, total_params=compute_objective(auc)
    # nni.report_final_result(auc)  
    nni.report_final_result(metric_auc_flops)  

    # save the model  
    # Convert the model’s state_dict to a dictionary and save with joblib
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dict = model.state_dict()
    joblib.dump(model_dict, f"models/EUC2/{timestamp}_model_{nni.get_experiment_id()}_{nni.get_trial_id()}.joblib") 

    return flops, total_params


def adjust_weight_decay(optimizer, epoch, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] *= decay_rate


def inference_on_test(model, test_loader, test_data_name):

    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs).squeeze() 
            all_labels.extend(labels.cpu().numpy())  
            all_preds.extend(probs.cpu().numpy())

   
        # Convert probabilities to binary predictions
    all_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]


    auc = roc_auc_score(all_labels, all_preds_binary)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary, zero_division=0)
    recall = recall_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary)

    f1_euc_mlp, t1_euc_mlp, th1=roc_curve(all_labels, all_preds_binary, pos_label=1)
      
    CM = confusion_matrix(all_labels, all_preds_binary)

    TN_euc_mlp = CM[0][0]
    FN_euc_mlp = CM[1][0]
    TP_euc_mlp = CM[1][1]
    FP_euc_mlp = CM[0][1]
        
    clf_tpr=(TP_euc_mlp/(TP_euc_mlp+FN_euc_mlp))
    clf_fpr=(FP_euc_mlp/(FP_euc_mlp+TN_euc_mlp)) 
    


    if test_data_name==" test data " or " validation data ":
        print(f"*********** Results of inference on {test_data_name} ********")
        print(f" > AUC: {auc:.4f}")
        print(f" > TPR: {t1_euc_mlp}, FPR: {f1_euc_mlp}")
        print(f" > TPR: {clf_tpr}")
        print(f" > FPR: {clf_fpr}")
        print(f" > Accuracy: {accuracy:.4f}")
        print(f" > Precision: {precision:.4f}")
        print(f" > Recall: {recall:.4f}")
        print(f" > F1 Score: {f1:.4f} \n \n ")

        print(torch.cuda.memory_allocated())  
        print(torch.cuda.memory_reserved()) 

    # metrics = {
    #     'AUC': auc,
    #     'Accuracy': accuracy,
    #     'Precision': precision,
    #     'Recall': recall,
    #     'F1 Score': f1
    # } >> doesn't work :/

    # nni.report_intermediate_result(metrics)  # Report intermediate metrics (if in training loop)
    
    # Return the metrics if needed
    return auc, all_preds_binary, all_labels


def compute_objective(auc, input_size=2, max_flops=134145, alpha=0.9, beta=0.1):

   
    input_shape = (input_size,)  

    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Alpha : {alpha}")
    print(f"Beta : {beta}")
    print("FLOPs:", flops)
    print("Parameters:", params)
    print("Parameters 2 :", total_params)
   
    normalized_flops = flops / max_flops
    print(f"normalized flops: {normalized_flops}")
    print(f"AUC: {auc}")
    objective_value = alpha * auc - beta * normalized_flops
    return objective_value, flops, total_params


def min_max_normalize(train_array, save_factors_path):

    min_vals = np.min(train_array, axis=0)
    max_vals = np.max(train_array, axis=0)
    print(f"Min: {min_vals}")
    print(f"Max: {max_vals}")
    normalized_array = (train_array - min_vals) / (max_vals - min_vals)
    
    
    return normalized_array, min_vals, max_vals



def std_normalize(train_array, save_factors_path):
    mean = np.mean(train_array, axis=0)
    std = np.std(train_array, axis=0)
    
    print(f"Mean: {mean}")
    print(f"STD: {std}")
    std[std == 0] = 1

    normalized_array = (train_array - mean) / std
    
    return normalized_array, mean, std


def normalize_test_array(test_array, use_standardization, fac1, fac2, load_factors_path):
    if not use_standardization:
        max_vals = fac1 
        min_vals = fac2 
        normalized_array = (test_array - min_vals) / (max_vals - min_vals)
    
    else:
        mean_vals = fac1 
        std_vals = fac2 
        std_vals[std_vals == 0] = 1
        normalized_array = (test_array - mean_vals) / std_vals
    
    return normalized_array


def calculate_inference_time(model, dataset, single_instance=False, device="cpu"):
    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)


    with torch.no_grad():
        start_time = time.perf_counter() 
        for inputs in data_iter:
            inputs = inputs[0].to(device)  
            _ = model(inputs)
            if single_instance:
                break
        end_time = time.perf_counter() 

    return end_time - start_time



def calculate_gpu_inference_time(model, dataset, single_instance=False, device="cuda"):

    if device != "cuda":
        raise ValueError("CUDA inference requires device to be 'cuda'.")

    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for inputs in data_iter:
            inputs = inputs[0].to(device)
            _ = model(inputs)
            if single_instance:
                break
        end_event.record()


    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event) / 1000  

    return inference_time



if __name__ == "__main__":
    params = get_next_parameter()
    parser = argparse.ArgumentParser(description='Train distance model with different configurations')
    parser.add_argument('--distance', type=str, default='euc', help='Distance to use, euc or dtw')
    args = parser.parse_args()
    distance_type = args.distance


    data_path="/data/dust/user/boukela/code/Pcode_dev/Data"
    #read data
    Xtrain, Xtest, Ytrain, Ytest =read_data(data_path)
        
    if distance_type=='euc':
        train_x=Xtrain[:, 0:2]
        test_x=Xtest[:, 0:2]
    elif distance_type=='dtw':
        train_x=Xtrain[:, 2:]
        test_x=Xtest[:, 2:]
    elif distance_type=='both':
        train_x=Xtrain
        test_x=Xtest
    else:
        print("distance incorrect")

    print("\n ***** Normalization ***** \n")
    train_x_norm, means, stds = std_normalize(train_x, 'normalization_factors.pkl')
    train_x_norm2, mins, maxs = min_max_normalize(train_x, 'normalization_factors.pkl')

    test_x_norm = normalize_test_array(test_x, True, means, stds, 'normalization_factors.pkl')
    test_x_norm2 = normalize_test_array(test_x, False, maxs, mins, 'normalization_factors.pkl')

    train_tensor = torch.tensor(train_x_norm, dtype=torch.float32) 
    Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.float32).unsqueeze(1) 
    
    test_tensor = torch.tensor(test_x_norm, dtype=torch.float32)
    Ytest_tensor = torch.tensor(Ytest, dtype=torch.float32).unsqueeze(1)

    train_dataset1 = TensorDataset(train_tensor, Ytrain_tensor)
    test_dataset = TensorDataset(test_tensor, Ytest_tensor)

    train_size = int(0.9 * len(train_dataset1)) 
    val_size = len(train_dataset1) - train_size 

    train_dataset, val_dataset = random_split(train_dataset1, [train_size, val_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    positive_count = 0
    negative_count = 0

    for _, labels in val_loader:
        positive_count += (labels == 1).sum().item() 
        negative_count += (labels == 0).sum().item() 

    print(f"Positive examples in validation dataset: {positive_count}")
    print(f"Negative examples in validation dataset: {negative_count}")

    input_size = 2  
    output_size = 1   

    torch.manual_seed(seed)
    model = MLP(
        input_size,
        params["hidden_layer_1_neurons"],
        params["hidden_layer_2_neurons"],
        params["hidden_layer_3_neurons"],
        output_size
    )

    model = model.to(device) 

    print("-------------- Model --------------")
    summary(model, input_size=(2,))
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            num_neurons = layer.out_features if hasattr(layer, 'out_features') else layer.out_channels
            print(f"Layer: {name}, Type: {layer.__class__.__name__}, Neurons: {num_neurons}")


    class_counts = Counter()
    for _, targets in train_loader:
        class_counts.update(targets.view(-1).tolist())
    pos_weight = torch.tensor([class_counts[0]/ class_counts[1]]) 
    pos_weight = pos_weight.to(device)
    print("\n****** class counts ***** ")
    print(f"****** num_negative : {class_counts[0]}, num_positive : {class_counts[1]} ***** \n\n")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 100
    flops, total_params= train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, epochs)


    cpu_time_single = calculate_inference_time(model, test_loader, single_instance=True, device="cpu")
    # time_full = calculate_inference_time(model, dataset, single_instance=False, device="cuda")
    gpu_time_single = calculate_gpu_inference_time(model, test_loader, single_instance=True, device="cuda")
    # gpu_time_full = calculate_gpu_inference_time(model, dataset, single_instance=False, device="cuda")



    # evaluate the model on the test dataset
    metrics, preds, all_labels = inference_on_test(model, test_loader, " test data ")
    print(f" \n\n************ final AUC ************ : {metrics}")

    y_pred_euc_mlp=preds
    print(np.unique(np.array(y_pred_euc_mlp)))
    # Y_test = keras.utils.to_categorical(test[:,4], 2)

    f1_euc_mlp, t1_euc_mlp, th1=roc_curve(all_labels, y_pred_euc_mlp, pos_label=1)
    auc_euc_mlp = auc(f1_euc_mlp, t1_euc_mlp)
    # print(f"auc={auc(f1_euc_mlp, t1_euc_mlp)}, TPR= {t1_euc_mlp}, FPR= {f1_euc_mlp}")
    accuracy = accuracy_score(all_labels, y_pred_euc_mlp)
    print(f" > Accuracy on the test set: {accuracy}")
    CM = confusion_matrix(all_labels, y_pred_euc_mlp)

    TN_euc_mlp = CM[0][0]
    FN_euc_mlp = CM[1][0]
    TP_euc_mlp = CM[1][1]
    FP_euc_mlp = CM[0][1]
        
    clf_tpr=(TP_euc_mlp/(TP_euc_mlp+FN_euc_mlp))
    clf_fpr=(FP_euc_mlp/(FP_euc_mlp+TN_euc_mlp)) 
    print(f" > TPR: {clf_tpr}")
    print(f" > FPR: {clf_fpr}")

    # save results
evaluation_results = {
"Distance": [distance_type],
"Neurons": [0],
"Epochs": [epochs],
"Accuracy": [accuracy],
"AUC": [auc(f1_euc_mlp, t1_euc_mlp)],
"TPR": [clf_tpr],
"FPR": [clf_fpr],
"Flops": [flops],
"NumParams": [total_params],
"cpu_time_single": [cpu_time_single],
"gpu_time_single": [gpu_time_single]
}
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"./models/EUC2/{timestamp}-eval-distance-based-{distance_type}-epochs-{epochs}-Experiment-{nni.get_experiment_id()}_{nni.get_trial_id()}-withoutInit.csv"

pd.DataFrame(evaluation_results).to_csv(filename)
        # Report final result to NNI
        # nni.report_final_result(metrics)