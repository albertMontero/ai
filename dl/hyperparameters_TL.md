## 1. **Learning Rate (LR)**

The single most important hyperparameter for transfer learning.

* **If freezing most layers** (only training the final classifier):
  Use a relatively higher LR (e.g., `1e-3` or `1e-4`) — since only a few parameters are updated.

* **If fine-tuning the entire network** (unfreezing all layers):
  Use a much smaller LR (e.g., `1e-5` to `1e-6`) — pretrained weights are delicate and can easily be destroyed by large updates.

* **Tip:**
  Use **differential learning rates**, e.g., smaller for backbone layers, larger for the final classifier:

  ```python
  optimizer = torch.optim.Adam([
      {'params': model.backbone.parameters(), 'lr': 1e-5},
      {'params': model.fc.parameters(), 'lr': 1e-3}
  ])
  ```

---

## 2. **Optimizer**

Controls how weights are updated.

* Common choices:

    * `Adam` → fast convergence, often good default.
    * `SGD` + momentum (e.g., 0.9) → slower but sometimes better generalization.
    * `AdamW` → similar to Adam but with decoupled weight decay (recommended for fine-tuning).

* Example:

  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
  ```

---

## 3. **Weight Decay (L2 Regularization)**

Helps prevent overfitting on small datasets.

* Typical range: `1e-4` to `1e-6`.
* Too large → underfitting; too small → overfitting risk.

---

## 4. **Batch Size**

Affects both convergence and generalization.

* Depends on GPU memory. Common values: `16`, `32`, `64`.
* Larger batches → faster but **might require a higher LR**.
* Smaller batches → slower but better generalization.

---

## 5. **Number of Epochs**

Pretrained models converge faster.

* For transfer learning: **5–20 epochs** often sufficient.
* For full fine-tuning: **20–50 epochs**, with LR scheduling and early stopping.

---

## 6. **Learning Rate Scheduler**

Helps maintain stable training and improve convergence.

Common choices:

* `ReduceLROnPlateau` → decreases LR when validation loss stops improving.
* `StepLR` → decays LR every few epochs.
* `CosineAnnealingLR` → smoothly reduces LR over time.

Example:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
```

---

## 7. **Layer Freezing / Unfreezing Strategy**

This is a *crucial* hyperparameter in transfer learning.

* **Feature extraction:** freeze all convolutional layers:

  ```python
  for param in model.parameters():
      param.requires_grad = False
  ```

  Then train only the final `fc` layer.

* **Fine-tuning:** unfreeze later blocks gradually:

  ```python
  for name, param in model.named_parameters():
      if "layer4" in name or "fc" in name:
          param.requires_grad = True
  ```

  This allows the model to adapt high-level features to the new dataset.

---

## 8. **Data Augmentation Parameters**

Although not part of the model itself, they strongly affect performance.

* Rotation, flipping, color jitter, random crop, normalization.
* `torchvision.transforms` is your go-to tool:

  ```python
  transforms = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  ```

---

## 9. **Dropout / Regularization Layers**

If your dataset is small, consider adding or modifying dropout in the classifier head:

```python
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
```

Hyperparameter: dropout rate (e.g., 0.3–0.5).

---

## 10. **Loss Function**

Usually fixed, but worth mentioning:

* `nn.CrossEntropyLoss()` → standard for single-label classification.
* `nn.BCEWithLogitsLoss()` → for multi-label problems.
* Weighted variants if classes are imbalanced.

---

## 11. **Early Stopping Criteria**

* Monitor **validation loss** or **accuracy**.
* Stop training if no improvement after `patience` epochs (e.g., 5).

---

### 🔧 Summary Table

| Category       | Hyperparameter            | Typical Range / Options             | Importance |
| -------------- | ------------------------- | ----------------------------------- | ---------- |
| Training       | Learning rate             | 1e-3 → 1e-6                         | ⭐⭐⭐⭐⭐      |
| Training       | Optimizer                 | Adam / AdamW / SGD+Momentum         | ⭐⭐⭐⭐       |
| Training       | Weight decay              | 1e-4 → 1e-6                         | ⭐⭐⭐        |
| Training       | Batch size                | 16 → 64                             | ⭐⭐⭐        |
| Training       | Epochs                    | 5 → 50                              | ⭐⭐         |
| Strategy       | Layers to freeze/unfreeze | depends on dataset size             | ⭐⭐⭐⭐       |
| Regularization | Dropout rate              | 0.3 → 0.5                           | ⭐⭐         |
| Augmentation   | Transform parameters      | dataset dependent                   | ⭐⭐⭐        |
| Scheduler      | LR schedule type          | StepLR / ReduceLROnPlateau / Cosine | ⭐⭐⭐        |

