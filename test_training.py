
import torch
import torch.nn as nn
from torchvision import models

print("1. Starting test...", flush=True)

torch.set_num_threads(2)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 102)

print("2. Model created", flush=True)

x = torch.randn(4, 3, 224, 224)
y = torch.randint(0, 102, (4,))

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("3. Starting forward pass...", flush=True)

output = model(x)
loss = criterion(output, y)

print("4. Starting backward pass...", flush=True)

loss.backward()
optimizer.step()

print("5. SUCCESS! Training works.", flush=True)
