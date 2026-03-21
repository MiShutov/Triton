from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def preprocess_mnist_data(examples):
    images = []
    labels = []
    original_dim = 28 * 28
    new_dim = (original_dim // 128) * 128
    
    for img, label in zip(examples['image'], examples['label']):
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = img_array.reshape(-1)
        images.append(img_array[:new_dim])
        labels.append(label)
    # print(images[0].shape)
    return {'image': images, 'label': labels}


def prepare_mnist_dataset_for_training(batch_size):
    dataset = load_dataset("ylecun/mnist")
    dataset = dataset.map(preprocess_mnist_data, batched=True)
    dataset.set_format(type='torch', columns=['image', 'label'])

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def train_model(
        model,
        optimizer,
        train_loader,
        loss_fn,
        num_epochs=10,
        autocast_dtype=None,
    ):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    out = model(x)
                    loss = loss_fn(out, y)
            else:
                out = model(x)
                loss = loss_fn(out, y)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss


@torch.no_grad()
def test_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            out = model(x)
            _, predicted = torch.max(out, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%\n")
    return test_accuracy
