import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class NoForgettingLearning:
    """
    Implementation of No Forgetting Learning (NFL) algorithm for Continual Learning.
    NFL avoids accessing data from previous tasks and uses network parameters
    trained on earlier tasks to prevent catastrophic forgetting.
    """
    def __init__(self, model, device='cuda', lr=0.01, batch_size=128, 
                 epochs_per_step=50, temperature=2.0, lambda_val=1.0, 
                 omega=1.0, beta=1.0, alpha=0.5):
        """
        Initialize NFL with a base model and hyperparameters.
        
        Args:
            model: Base neural network model (e.g., ResNet-18)
            device: Device to run computations on ('cuda' or 'cpu')
            lr: Learning rate for SGD optimizer
            batch_size: Batch size for training
            epochs_per_step: Number of epochs for each step of NFL
            temperature: Temperature parameter for knowledge distillation
            lambda_val: Hyperparameter λ for balancing CE and KD losses in step 3
            omega: Hyperparameter ω for balancing CE and KD losses in step 4
            beta: Hyperparameter β for balancing CE and KD losses in step 5
            alpha: Hyperparameter α for weighting KD losses in step 5
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs_per_step = epochs_per_step
        self.temperature = temperature
        self.lambda_val = lambda_val
        self.omega = omega
        self.beta = beta
        self.alpha = alpha
        
        # Store parameters from previous tasks
        self.theta_s = None  # Shared parameters (all parameters except the last layer)
        self.theta_t = {}    # Task-specific parameters (last layer for each task)
        
        # Current task ID
        self.current_task = 0

    def _extract_features(self, x):
        """Extract features using shared parameters (all layers except the last)"""
        # This should be implemented based on the specific model architecture
        # For ResNet, this would be all layers before the final FC layer
        features = self.model.feature_extractor(x)
        return features
        
    def _apply_classifier(self, features, task_id):
        """Apply task-specific classifier to features"""
        # This should use the task-specific classifier for the given task
        return self.theta_t[task_id](features)
    
    def _kd_loss(self, logits, targets, temperature=2.0):
        """
        Knowledge Distillation loss function
        
        Args:
            logits: Current model's outputs before softmax
            targets: Soft targets (teacher's outputs)
            temperature: Temperature parameter for softening distributions
        """
        log_pred = F.log_softmax(logits / temperature, dim=1)
        soft_targets = F.softmax(targets / temperature, dim=1)
        return F.kl_div(log_pred, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    def train_first_task(self, train_loader, task_id=0):
        """
        Train the model on the first task (Step 1)
        
        Args:
            train_loader: DataLoader for the first task
            task_id: ID for the first task
        """
        print(f"Training Task {task_id} (Step 1)")
        
        # Initialize optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_step):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            print(f'Epoch {epoch+1}/{self.epochs_per_step}, Loss: {running_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
        
        # Store parameters after training
        self.current_task = task_id
        
        # Store shared parameters (all except last layer)
        self.theta_s = copy.deepcopy(self.model.state_dict())
        
        # Remove last layer parameters from theta_s
        for name in list(self.theta_s.keys()):
            if 'fc' in name or 'classifier' in name:  # This depends on the model architecture
                del self.theta_s[name]
        
        # Store task-specific parameters (last layer)
        task_classifier = copy.deepcopy(self.model.fc)  # This should be adapted for your model
        self.theta_t[task_id] = task_classifier
        
        print(f"Task {task_id} training completed and parameters stored")
    
    def generate_soft_targets(self, data_loader):
        """
        Generate soft targets for the new task using the previous model (Step 1)
        
        Args:
            data_loader: DataLoader for the new task
        """
        print("Generating soft targets for the new task")
        
        soft_targets = []
        all_inputs = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                # Get logits (before softmax)
                features = self._extract_features(inputs)
                logits = self._apply_classifier(features, self.current_task)
                
                soft_targets.append(logits.cpu())
                all_inputs.append(inputs.cpu())
                all_labels.append(labels)
        
        # Combine all batches
        soft_targets = torch.cat(soft_targets, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_inputs, all_labels, soft_targets
    
    def train_step2(self, train_loader, task_id):
        """
        Train only the last layer for the new task while freezing 
        the shared parameters (Step 2)
        
        Args:
            train_loader: DataLoader for the new task
            task_id: ID for the new task
        """
        print(f"Training new task classifier with frozen feature extractor (Step 2)")
        
        # Create a new model with the same architecture
        new_model = copy.deepcopy(self.model)
        
        # Load the shared parameters
        for name, param in new_model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:  # Freeze all except last layer
                param.requires_grad = False
        
        # Initialize optimizer (only for the last layer parameters)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, new_model.parameters()), 
                            lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        new_model.train()
        for epoch in range(self.epochs_per_step):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = new_model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            print(f'Epoch {epoch+1}/{self.epochs_per_step}, Loss: {running_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
        
        # Store the new task-specific parameters
        self.theta_t[task_id] = copy.deepcopy(new_model.fc)  # This should be adapted for your model
        
        print(f"Task {task_id} classifier training completed")
        return new_model
    
    def train_step3(self, train_loader, soft_targets, task_id):
        """
        Train a new model with frozen task-specific parameters for the new task
        while learning shared parameters to preserve old task knowledge (Step 3)
        
        Args:
            train_loader: DataLoader for the new task
            soft_targets: Soft targets generated from the previous model
            task_id: ID for the new task
        """
        print(f"Training shared parameters with knowledge distillation (Step 3)")
        
        # Create a new model with the same architecture
        new_model = copy.deepcopy(self.model)
        
        # Replace the task-specific layer for the new task
        new_model.fc = self.theta_t[task_id]
        
        # Make old classifier available for old task outputs
        old_classifier = self.theta_t[self.current_task]
        
        # Make only shared parameters trainable
        for name, param in new_model.named_parameters():
            if 'fc' in name or 'classifier' in name:  # Freeze classifier layers
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Initialize optimizer (only for shared parameters)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, new_model.parameters()), 
                            lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        new_model.train()
        for epoch in range(self.epochs_per_step):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get corresponding soft targets for this batch
                batch_soft_targets = soft_targets[i*self.batch_size:
                                                (i+1)*self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Extract features
                features = new_model.feature_extractor(inputs)
                
                # Apply old task classifier to get old task outputs
                old_task_outputs = old_classifier(features)
                
                # Apply new task classifier to get new task outputs
                new_task_outputs = new_model.fc(features)
                
                # Calculate KD loss for old task 
                kd_loss = self._kd_loss(old_task_outputs, batch_soft_targets, self.temperature)
                
                # Calculate CE loss for new task
                ce_loss = criterion(new_task_outputs, targets)
                
                # Combined loss
                loss = kd_loss + self.lambda_val * ce_loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = new_task_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            print(f'Epoch {epoch+1}/{self.epochs_per_step}, Loss: {running_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
        
        # Store updated shared parameters
        updated_theta_s = new_model.state_dict()
        for name in list(updated_theta_s.keys()):
            if 'fc' in name or 'classifier' in name:
                del updated_theta_s[name]
        
        # Store updated task-specific parameters for the old task
        updated_old_classifier = old_classifier
        
        print(f"Step 3 completed - updated shared parameters and old task classifier")
        return updated_theta_s, updated_old_classifier
    
    def train_step4(self, train_loader, soft_targets, updated_theta_s, 
                   updated_old_classifier, task_id):
        """
        Fine-tune shared parameters along with the new task classifier (Step 4)
        
        Args:
            train_loader: DataLoader for the new task
            soft_targets: Soft targets generated from the previous model
            updated_theta_s: Updated shared parameters from Step 3
            updated_old_classifier: Updated classifier for the old task from Step 3
            task_id: ID for the new task
        """
        print(f"Fine-tuning shared parameters and new task classifier (Step 4)")
        
        # Create a new model with the same architecture
        new_model = copy.deepcopy(self.model)
        
        # Load updated shared parameters
        current_state_dict = new_model.state_dict()
        for name, param in updated_theta_s.items():
            current_state_dict[name] = param
        new_model.load_state_dict(current_state_dict, strict=False)
        
        # Set the task-specific layer for the new task
        new_model.fc = self.theta_t[task_id]
        
        # Set old classifier for knowledge distillation
        old_classifier = updated_old_classifier
        
        # Initialize optimizer for shared parameters and new task classifier
        optimizer = optim.SGD([
            {'params': [p for n, p in new_model.named_parameters() if 'fc' not in n]},
            {'params': new_model.fc.parameters(), 'lr': self.lr * 10}  # Higher LR for classifier
        ], lr=self.lr, momentum=0.9, weight_decay=5e-4)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        new_model.train()
        for epoch in range(self.epochs_per_step):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get corresponding soft targets for this batch
                batch_soft_targets = soft_targets[i*self.batch_size:
                                                (i+1)*self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Extract features
                features = new_model.feature_extractor(inputs)
                
                # Apply old task classifier to get old task outputs
                old_task_outputs = old_classifier(features)
                
                # Apply new task classifier to get new task outputs
                new_task_outputs = new_model.fc(features)
                
                # Calculate KD loss for old task 
                kd_loss = self._kd_loss(old_task_outputs, batch_soft_targets, self.temperature)
                
                # Calculate CE loss for new task
                ce_loss = criterion(new_task_outputs, targets)
                
                # Combined loss
                loss = kd_loss + self.omega * ce_loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = new_task_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            print(f'Epoch {epoch+1}/{self.epochs_per_step}, Loss: {running_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
        
        # Store fine-tuned shared parameters
        fine_tuned_theta_s = new_model.state_dict()
        for name in list(fine_tuned_theta_s.keys()):
            if 'fc' in name or 'classifier' in name:
                del fine_tuned_theta_s[name]
        
        # Store fine-tuned new task classifier
        fine_tuned_new_classifier = copy.deepcopy(new_model.fc)
        
        print(f"Step 4 completed - fine-tuned shared parameters and new task classifier")
        return fine_tuned_theta_s, fine_tuned_new_classifier
    
    def calculate_new_logits(self, train_loader, fine_tuned_theta_s):
        """
        Calculate new logits for the old task using fine-tuned shared parameters (Step 5)
        
        Args:
            train_loader: DataLoader for the new task
            fine_tuned_theta_s: Fine-tuned shared parameters from Step 4
        """
        print(f"Calculating new logits with fine-tuned feature extractor")
        
        # Create a new model with the same architecture
        new_model = copy.deepcopy(self.model)
        
        # Load fine-tuned shared parameters
        current_state_dict = new_model.state_dict()
        for name, param in fine_tuned_theta_s.items():
            current_state_dict[name] = param
        new_model.load_state_dict(current_state_dict, strict=False)
        
        # Set the original task-specific layer for the old task
        old_classifier = self.theta_t[self.current_task]
        
        new_logits = []
        
        # Calculate new logits
        new_model.eval()
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(self.device)
                
                # Extract features using fine-tuned feature extractor
                features = new_model.feature_extractor(inputs)
                
                # Apply original classifier to get new logits
                logits = old_classifier(features)
                
                new_logits.append(logits.cpu())
        
        # Combine all batches
        new_logits = torch.cat(new_logits, dim=0)
        
        print(f"New logits calculation completed")
        return new_logits
    
    def train_step5(self, train_loader, soft_targets, new_logits, fine_tuned_theta_s, 
                   updated_old_classifier, fine_tuned_new_classifier, task_id):
        """
        Final training step with two knowledge distillation targets (Step 5)
        
        Args:
            train_loader: DataLoader for the new task
            soft_targets: Original soft targets from Step 1
            new_logits: New logits calculated in Step 5 calculation
            fine_tuned_theta_s: Fine-tuned shared parameters from Step 4
            updated_old_classifier: Updated old task classifier from Step 3
            fine_tuned_new_classifier: Fine-tuned new task classifier from Step 4
            task_id: ID for the new task
        """
        print(f"Final training step with dual knowledge distillation (Step 5)")
        
        # Create a new model with the same architecture
        new_model = copy.deepcopy(self.model)
        
        # Load fine-tuned shared parameters
        current_state_dict = new_model.state_dict()
        for name, param in fine_tuned_theta_s.items():
            current_state_dict[name] = param
        new_model.load_state_dict(current_state_dict, strict=False)
        
        # Original classifier for the old task (frozen)
        original_classifier = self.theta_t[self.current_task]
        
        # Updated classifier for the old task from Step 3 (trainable)
        old_classifier = updated_old_classifier
        
        # New task classifier (trainable)
        new_model.fc = fine_tuned_new_classifier
        
        # Prepare optimizable parameters
        params_to_optimize = [
            {'params': [p for n, p in new_model.named_parameters() if 'fc' not in n]},  # Shared params
            {'params': old_classifier.parameters(), 'lr': self.lr * 5},  # Old task classifier
            {'params': new_model.fc.parameters(), 'lr': self.lr * 10}  # New task classifier
        ]
        
        optimizer = optim.SGD(params_to_optimize, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        new_model.train()
        for epoch in range(self.epochs_per_step):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get corresponding soft targets and new logits for this batch
                batch_soft_targets = soft_targets[i*self.batch_size:
                                                (i+1)*self.batch_size].to(self.device)
                batch_new_logits = new_logits[i*self.batch_size:
                                            (i+1)*self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Extract features
                features = new_model.feature_extractor(inputs)
                
                # Apply original classifier to get outputs for first KD loss
                original_outputs = original_classifier(features)
                
                # Apply updated old classifier to get outputs for second KD loss
                updated_old_outputs = old_classifier(features)
                
                # Apply new task classifier to get new task outputs
                new_task_outputs = new_model.fc(features)
                
                # Calculate first KD loss using original soft targets
                kd_loss1 = self._kd_loss(original_outputs, batch_soft_targets, self.temperature)
                
                # Calculate second KD loss using new logits
                kd_loss2 = self._kd_loss(updated_old_outputs, batch_new_logits, self.temperature)
                
                # Calculate CE loss for new task
                ce_loss = criterion(new_task_outputs, targets)
                
                # Combined loss
                loss = self.alpha * kd_loss1 + (1 - self.alpha) * kd_loss2 + self.beta * ce_loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = new_task_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            print(f'Epoch {epoch+1}/{self.epochs_per_step}, Loss: {running_loss/len(train_loader):.4f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
        
        # Store the final parameters
        final_theta_s = new_model.state_dict()
        for name in list(final_theta_s.keys()):
            if 'fc' in name or 'classifier' in name:
                del final_theta_s[name]
        
        # Update shared parameters for next task
        self.theta_s = final_theta_s
        
        # Update task-specific parameters
        self.theta_t[self.current_task] = old_classifier
        self.theta_t[task_id] = copy.deepcopy(new_model.fc)
        
        # Update current task ID
        self.current_task = task_id
        
        # Update the model with the final parameters
        self._update_model()
        
        print(f"NFL training completed for task {task_id}")
    
    def _update_model(self):
        """Update the model with the current shared and task-specific parameters"""
        # Load shared parameters
        current_state_dict = self.model.state_dict()
        for name, param in self.theta_s.items():
            current_state_dict[name] = param
        
        # Load parameters for the current task
        self.model.fc = self.theta_t[self.current_task]
        
        self.model.load_state_dict(current_state_dict, strict=False)
    
    def train_next_task(self, train_loader, task_id):
        """
        Train the model on a new task using NFL algorithm
        
        Args:
            train_loader: DataLoader for the new task
            task_id: ID for the new task
        """
        print(f"=== Training Task {task_id} using NFL ===")
        
        # Step 1: Generate soft targets for the new task
        inputs, labels, soft_targets = self.generate_soft_targets(train_loader)
        
        # Create a new data loader with the same inputs and labels
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        new_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Step 2: Train only the last layer for the new task
        self.train_step2(new_loader, task_id)
        
        # Step 3: Train shared parameters with knowledge distillation
        updated_theta_s, updated_old_classifier = self.train_step3(
            new_loader, soft_targets, task_id
        )
        
        # Step 4: Fine-tune shared parameters and new task classifier
        fine_tuned_theta_s, fine_tuned_new_classifier = self.train_step4(
            new_loader, soft_targets, updated_theta_s, updated_old_classifier, task_id
        )
        
        # Step 5a: Calculate new logits using fine-tuned shared parameters
        new_logits = self.calculate_new_logits(new_loader, fine_tuned_theta_s)
        
        # Step 5b: Final training with dual knowledge distillation
        self.train_step5(
            new_loader, soft_targets, new_logits, fine_tuned_theta_s,
            updated_old_classifier, fine_tuned_new_classifier, task_id
        )
        
        print(f"=== Completed training for Task {task_id} ===")
    
    def eval_task(self, test_loader, task_id):
        """
        Evaluate the model on a specific task
        
        Args:
            test_loader: DataLoader for the task to evaluate
            task_id: ID of the task to evaluate
        """
        # Load appropriate task-specific parameters
        current_fc = self.model.fc
        self.model.fc = self.theta_t[task_id]
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Task {task_id} Accuracy: {accuracy:.2f}%')
        
        # Restore current task parameters
        self.model.fc = current_fc
        
        return accuracy


# Usage Example with ResNet-18
class FeatureExtractor(nn.Module):
    """
    Feature extractor module for the ResNet-18 model
    """
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        # Flatten the output for the FC layer
        return torch.flatten(x, 1)


class ResNetCL(nn.Module):
    """
    ResNet model adapted for continual learning
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNetCL, self).__init__()
        # Load pretrained ResNet-18 model
        import torchvision.models as models
        model = models.resnet18(pretrained=pretrained)
        
        # Feature extractor (all layers except the last FC layer)
        self.feature_extractor = FeatureExtractor(model)
        
        # Task-specific fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.fc(features)
        return output


# Example of how to use the NFL implementation
def nfl_example():
    # Create ResNet model for continual learning
    model = ResNetCL(num_classes=10)
    
    # Create NFL instance
    nfl = NoForgettingLearning(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.01,
        batch_size=128,
        epochs_per_step=50,
        temperature=2.0,
        lambda_val=1.0,
        omega=1.0,
        beta=1.0,
        alpha=0.5
    )
    
    # Example: Train on the first task (e.g., CIFAR-10)
    # train_loader_task0 = ...  # DataLoader for task 0
    # nfl.train_first_task(train_loader_task0, task_id=0)
    
    # Example: Train on the second task
    # train_loader_task1 = ...  # DataLoader for task 1
    # nfl.train_next_task(train_loader_task1, task_id=1)
    
    # Example: Evaluate on specific tasks
    # test_loader_task0 = ...  # DataLoader for task 0 test set
    # accuracy_task0 = nfl.eval_task(test_loader_task0, task_id=0)
    
    # test_loader_task1 = ...  # DataLoader for task 1 test set
    # accuracy_task1 = nfl.eval_task(test_loader_task1, task_id=1)
    
    # print(f"Average accuracy: {(accuracy_task0 + accuracy_task1) / 2:.2f}%")

if __name__ == "__main__":
    nfl_example()