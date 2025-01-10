import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process import HeatmapGenerator, HeatmapToKeypointsConverter
from utils import calculate_precision_recall, calculate_mAP, calculate_oks, calculate_pck, plot_metrics
from tqdm import tqdm


def train_model(model, dataset, batch_size=32, num_epochs=10, learning_rate=1e-3, device='cuda', patience=3, weight_decay=1e-5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Split dataset into train and validation
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    best_val_loss = float('inf')
    stop_counter = 0  # Counter for early stopping

    train_losses = []
    val_losses = []
    pck_values = []
    oks_values = []
    precision_values = []
    recall_values = []
    mAP_values = []
    prev_lr = optimizer.param_groups[0]['lr']  # Get initial learning rate


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_oks = 0.0
        total_pck = 0.0
        total_precision = 0.0
        total_recall = 0.0

        precision_list = []
        recall_list = []

        heatmap2Keypoints = HeatmapToKeypointsConverter()       
        heatmap = HeatmapGenerator(64, 16)

        train_dataloader_iter = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for i, data in enumerate(train_dataloader_iter):
            images, keypoints = data['image'].to(device), data['keypoints'].to(device)
            target = torch.from_numpy(heatmap(keypoints)).to(device)

            optimizer.zero_grad()

            keypoints_pred = model(images)

            keypoints_pred = torch.stack(keypoints_pred, dim=0)
            for j in range(len(keypoints_pred)):
                loss = criterion(keypoints_pred[j].to(torch.float32), target.to(torch.float32))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            predKeyPoints = heatmap2Keypoints(heatmap_grid=keypoints_pred).to(device)

            # Calculate metrics
            oks = calculate_oks(predKeyPoints, keypoints)
            pck = calculate_pck(predKeyPoints, keypoints)
            precision, recall = calculate_precision_recall(predKeyPoints, keypoints)
            total_oks += oks.item()
            total_pck += pck.item()
            total_precision += precision
            total_recall += recall

            precision_list.append(precision)
            recall_list.append(recall)

            train_dataloader_iter.set_postfix({'Loss': running_loss / (i + 1),
                                               'PCK': total_pck / (i + 1),
                                               'OKS': total_oks / (i + 1),
                                               'Precision': total_precision / (i + 1),
                                               'Recall': total_recall / (i + 1)})

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        pck_values.append(total_pck / len(train_dataloader))
        oks_values.append(total_oks / len(train_dataloader))
        precision_values.append(total_precision / len(train_dataloader))
        recall_values.append(total_recall / len(train_dataloader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                images, keypoints = data['image'].to(device), data['keypoints'].to(device)
                target = torch.from_numpy(heatmap(keypoints)).to(device)

                keypoints_pred = model(images)

                for i in range(len(keypoints_pred)):
                    val_loss += criterion(keypoints_pred[i].to(torch.float32), target.to(torch.float32)).item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Calculate mAP
        precision_tensor = torch.tensor(precision_list, device=device)
        recall_tensor = torch.tensor(recall_list, device=device)
        mAP_tensor = calculate_mAP(precision_tensor, recall_tensor)
        mAP = mAP_tensor.item()  # Convert tensor to float
        mAP_values.append(mAP)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.7f}, Val Loss: {avg_val_loss:.7f}, mAP: {mAP:.4f}')

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate

        # Print the learning rate only if it changes
        if current_lr != prev_lr:
            print("Learning rate changed to:", current_lr)
            prev_lr = current_lr  # Update previous learning rate

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
            best_val_loss = avg_val_loss
            stop_counter = 0
        else:
            stop_counter += 1

        # Early stopping
        if stop_counter >= patience:
            print("Early stopping...")
            break

    # Plot

    plot_metrics(train_losses, val_losses, pck_values, oks_values, precision_values, recall_values, mAP_values)

    print("\n---Finished Training---\n")
