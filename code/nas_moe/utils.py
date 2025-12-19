import random
import numpy as np
import io
import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def plot_single_cell(arch_dict, cell_name):
    g = graphviz.Digraph(
        node_attr=dict(style='filled', shape='rect', align='center'),
        format='png'
    )
    g.body.extend(['rankdir=LR'])

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(arch_dict) % 2 == 0

    for i in range(2, 6):
        g.node(str(i), fillcolor='lightblue')

    for i in range(2, 6):
        for j in range(2):
            op = arch_dict[f'{cell_name}/op_{i}_{j}']
            from_ = arch_dict[f'{cell_name}/input_{i}_{j}'][0]
            if from_ == 0:
                u = 'c_{k-2}'
            elif from_ == 1:
                u = 'c_{k-1}'
            else:
                u = str(from_)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')

    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(2, 6):
        g.edge(str(i), 'c_{k}', fillcolor='gray')

    g.attr(label=f'{cell_name.capitalize()} cell')

    image = Image.open(io.BytesIO(g.pipe()))
    return image

def plot_double_cells(arch_dict):
    image1 = plot_single_cell(arch_dict, 'normal')
    image2 = plot_single_cell(arch_dict, 'reduce')
    height_ratio = max(image1.size[1] / image1.size[0], image2.size[1] / image2.size[0])
    _, axs = plt.subplots(1, 2, figsize=(20, 10 * height_ratio))
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28x1 -> 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x32 -> 14x14x64
        
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Первый сверточный блок
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Второй сверточный блок
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Полносвязные слои
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)

def evaluate_surrogate(surr: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu",
                       criterion: torch.nn.Module = None) -> float:
    """Вычисляет средний loss на loader (набор батчей Data)."""
    if criterion is None:
        criterion = torch.nn.MSELoss()
    surr.to(device)
    surr.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = surr(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def train_surrogate_with_val(surr: torch.nn.Module,
                             train_loader: torch.utils.data.DataLoader,
                             test_loader: torch.utils.data.DataLoader = None,
                             device: str = "cpu",
                             lr: float = 1e-3,
                             epochs: int = 50,
                             weight_decay: float = 0.0,
                             checkpoint_path: str = None,
                             verbose: bool = False) -> dict:
    """
    Тренировочный цикл для surrogate GNN с логированием train/val loss и отрисовкой графиков.

    Возвращает history = {'train': [...], 'test': [...]}.
    """
    optimizer = torch.optim.Adam(surr.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    history = {'train': [], 'test': []}

    surr.to(device)
    if verbose:
        iterator = range(1, epochs + 1)
    else:
        iterator = tqdm(range(1, epochs + 1), desc="Training surrogate")
    for epoch in iterator:
        surr.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = surr(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)
        test_loss = None
        if test_loader is not None:
            test_loss = evaluate_surrogate(surr, test_loader, device, criterion)

        history['train'].append(train_loss)
        history['test'].append(test_loss if test_loss is not None else float('nan'))

        if verbose:
            if test_loss is not None:
                print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")
            else:
                print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}")

        if checkpoint_path:
            torch.save({'epoch': epoch, 'model_state': surr.state_dict(),
                        'optimizer_state': optimizer.state_dict()}, checkpoint_path)

    # plt.figure(figsize=(8, 5))
    # plt.plot(history['train'], label='train')
    # if test_loader is not None:
    #     plt.plot(history['test'], label='test')
    # plt.xlabel('epoch')
    # plt.ylabel('MSE loss')
    # plt.title('Surrogate training')
    # plt.yscale('log')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return history


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    num_epochs: int,
    device: torch.device,
    scheduler=None,
    save_checkpoint: bool = True,
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
):
    
    # Создать директорию для чекпоинтов
    if save_checkpoint:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Инициализация истории
    history = {
        'train_losses': [],
        'test_losses': [],
        'train_accs': [],
        'test_accs': [],
        'best_test_acc': 0.0,
        'best_epoch': 0,
    }
    
    best_test_loss = float('inf')
    
    # Главный цикл обучения
    for epoch in range(num_epochs):
        # ==================== ОБУЧЕНИЕ ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            disable=not verbose
        )
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Статистика
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Обновить progressbar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Средние значения для эпохи
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_losses'].append(avg_train_loss)
        history['train_accs'].append(train_acc)
        
        # ==================== ВАЛИДАЦИЯ ====================
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        test_pbar = tqdm(
            test_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [Test]",
            disable=not verbose
        )
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * test_correct / test_total:.2f}%'
                })
        
        # Средние значения для эпохи
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        history['test_losses'].append(avg_test_loss)
        history['test_accs'].append(test_acc)
        
        # ==================== SCHEDULER ====================
        if scheduler is not None:
            scheduler.step()
            if verbose:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate: {current_lr:.6f}")
        
        # ==================== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ====================
        if save_checkpoint and avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            history['best_test_acc'] = test_acc
            history['best_epoch'] = epoch + 1
            
            checkpoint_path = f"{checkpoint_dir}/best_moe_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
            if verbose:
                print(f"✅ Best model saved to {checkpoint_path}")
        
        # ==================== ЛОГИРОВАНИЕ ====================
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {avg_test_loss:.6f} | Test Acc:  {test_acc:.2f}%")
            print("-" * 70)
    
    # === Загрузить лучшую модель ===
    if save_checkpoint:
        checkpoint_path = f"{checkpoint_dir}/best_moe_model.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if verbose:
            print(f"\n✅ Loaded best model from {checkpoint_path}")
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Training completed!")
        print(f"Best test accuracy: {history['best_test_acc']:.2f}% (Epoch {history['best_epoch']})")
        print(f"{'=' * 70}")
    
    return history


# ==================== ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ====================

def plot_training_history(
    history
):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='train')
    plt.plot(history['test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Surrogate training')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_training_summary(history):
    """Выводит сводку результатов обучения."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs: {len(history['train_losses'])}")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {history['train_losses'][-1]:.6f}")
    print(f"  Test Loss:  {history['test_losses'][-1]:.6f}")
    print(f"  Train Acc:  {history['train_accs'][-1]:.2f}%")
    print(f"  Test Acc:   {history['test_accs'][-1]:.2f}%")
    print(f"\nBest metrics:")
    print(f"  Best Test Acc:  {history['best_test_acc']:.2f}%")
    print(f"  Best Epoch:     {history['best_epoch']}")
    print("=" * 70 + "\n")
