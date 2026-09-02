import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import datetime

from dataset import build_dataloader
from model_ewc import CrowdContinualClassifier, EWCLifelongManager

VIRAT_CLASSES = {
    0: "Normal Pedestrian Motion",
    1: "Loitering / Lingering",
    2: "Carrying / Unloading Object",
    3: "Entering Vehicle",
    4: "Exiting Vehicle",
    5: "Unattended Package Drop"
}

def log_surveillance_alerts(model, dataloader, location_id, device, output_csv="surveillance_alerts.csv"):
    model.eval()
    records = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            logits, _ = model(x)
            probabilities = nn.functional.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            for i in range(len(x)):
                pred_idx = predictions[i].item()
                conf = confidences[i].item()
                risk = round(1.0 - conf if pred_idx in [1, 5] else conf * 0.4, 3)
                
                records.append({
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Location_ID": location_id,
                    "Detected_Condition": VIRAT_CLASSES.get(pred_idx, "Unknown"),
                    "Confidence_Score": round(conf, 4),
                    "Risk_Score": risk,
                    "Alert_Triggered": "YES" if risk > 0.55 else "NO"
                })

    df = pd.DataFrame(records)
    file_exists = os.path.exists(output_csv)
    df.to_csv(output_csv, mode='a', index=False, header=not file_exists)
    print(f"Logged {len(df)} detection records to '{output_csv}'.")

def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on device: {device}")

    tasks = ["task_1_scene0000", "task_2_scene0002", "task_3_scene0100"]
    available_tasks = [t for t in tasks if os.path.exists(os.path.join("dataset_tasks", t))]

    if not available_tasks:
        print("No task folders found in 'dataset_tasks/'. Please run '01_extract_frames.py' first.")
        return

    model = CrowdContinualClassifier(num_classes=len(VIRAT_CLASSES)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    cl_manager = EWCLifelongManager(model=model, ewc_lambda=350.0)

    retention_matrix = np.zeros((len(available_tasks), len(available_tasks)))
    all_dataloaders = []

    for task_idx, task_name in enumerate(available_tasks):
        task_dir = os.path.join("dataset_tasks", task_name)
        print(f"\n=================== TRAINING TASK {task_idx+1}: {task_name} ===================")
        
        train_loader = build_dataloader(task_dir, batch_size=16, is_train=True)
        all_dataloaders.append(train_loader)

        # Continual training phase
        model.train()
        for epoch in range(2):
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                logits, _ = model(x)
                loss = criterion(logits, y) + cl_manager.compute_ewc_loss()

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/2 | Loss: {running_loss/max(1, len(train_loader)):.4f}")

        # Post-task registration
        cl_manager.register_task(train_loader, task_id=task_idx, device=device)
        cl_manager.update_replay_buffer(train_loader)

        # Retention validation across all learned tasks
        model.eval()
        for past_idx in range(task_idx + 1):
            val_loader = all_dataloaders[past_idx]
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x)[0].argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)
            retention_matrix[task_idx, past_idx] = round(correct / max(1, total), 4)

        print(f"Current Retention Vector: {retention_matrix[task_idx, :task_idx+1]}")
        log_surveillance_alerts(model, train_loader, location_id=task_name, device=device)

    print("\nContinual learning pipeline completed.")
    print("Final Task Retention Matrix:")
    print(retention_matrix)

if __name__ == "__main__":
    run_pipeline()