import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from adaptive_chebyshev_class import AdaptiveChebyshev


def train_step(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        return loss

    loss = closure()
    optimizer.step(closure = closure)
    return loss.item()

torch.manual_seed(42)
model = nn.Linear(10, 1, bias=False)


model.weight.data.fill_(5.0)


data = torch.randn(100, 10)
target = torch.zeros(100, 1)

criterion = nn.MSELoss()


optimizer = AdaptiveChebyshev(
    model.parameters(),
    warm_up_steps=10,
    hvp_steps=1,
    exact_mu=True,
    my_heuristic_ratio= 0.01,
    epsilon=1e-5
)


loss_history = []
for epoch in range(50):
    loss_val = train_step(model, data, target, optimizer, criterion)
    loss_history.append(loss_val)


    for p in model.parameters():
        state = optimizer.state[p]
        L = state['lam_max']
        mu = state['lam_min']
        current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch:02d} | Loss: {loss_val:.6f} | L (max eigen): {L:.4f} | mu (min eigen): {mu:.4f} | Step size (eta): {current_lr:.6f}")


plt.plot(loss_history)
plt.yscale('log')
plt.title('Convergence of Adaptive Chebyshev Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log scale)')
plt.grid(True)
plt.show()