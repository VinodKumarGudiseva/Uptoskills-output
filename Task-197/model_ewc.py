import torch
import torch.nn as nn
import torchvision.models as models
import random

class CrowdContinualClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(CrowdContinualClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

class EWCLifelongManager:
    def __init__(self, model, ewc_lambda=400.0, replay_capacity=150):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.replay_capacity = replay_capacity
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.previous_tasks_params = {}
        self.fisher_matrices = {}
        self.replay_buffer = []

    def register_task(self, dataloader, task_id, device):
        """Calculates and stores the Fisher Information Matrix diagonals for a task."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        total_samples = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            logits, _ = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(x)
            total_samples += len(x)

        for n in fisher:
            fisher[n] /= max(1, total_samples)

        self.fisher_matrices[task_id] = fisher
        self.previous_tasks_params[task_id] = {
            n: p.clone().detach() for n, p in self.params.items()
        }

    def update_replay_buffer(self, dataloader):
        """Maintains memory replay vectors across deployment phases."""
        for x, y in dataloader:
            for i in range(len(x)):
                if len(self.replay_buffer) < self.replay_capacity:
                    self.replay_buffer.append((x[i].cpu(), y[i].cpu()))
                else:
                    idx = random.randint(0, len(self.replay_buffer) - 1)
                    self.replay_buffer[idx] = (x[i].cpu(), y[i].cpu())

    def compute_ewc_loss(self):
        """Calculates quadratic penalty against parameter shifts."""
        loss = 0.0
        for task_id in self.fisher_matrices:
            for n, p in self.model.named_parameters():
                if n in self.fisher_matrices[task_id]:
                    fisher = self.fisher_matrices[task_id][n]
                    mean = self.previous_tasks_params[task_id][n]
                    loss += (fisher * (p - mean).pow(2)).sum()
        return self.ewc_lambda * loss