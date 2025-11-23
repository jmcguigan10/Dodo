from torch.utils.data import DataLoader

model = ResidualFFIModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(num_epochs):
    model.train()
    for inv, resid_true, F_box_flat, F_true_flat in train_loader:
        inv = inv.to(device)
        F_box = F_box_flat.view(-1, 6, 4).to(device)
        F_true = F_true_flat.view(-1, 6, 4).to(device)

        optimizer.zero_grad()
        ΔF_pred_flat = model(inv)  # (B, 24)
        ΔF_pred = ΔF_pred_flat.view(-1, 6, 4)
        F_pred_raw = F_box + ΔF_pred
        F_pred = project_to_physical(F_pred_raw, F_init=None)

        loss = loss_fn(F_pred, F_true)
        loss.backward()
        optimizer.step()

    # validation step with no_grad + loss_fn(...)
    # update LR scheduler, check early stopping, etc.
