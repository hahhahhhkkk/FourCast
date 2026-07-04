model = fourcastmodel(6,512,8)
optimier = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
loss_fun = torch.nn.MSELoss()
epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimier,T_max=epochs, eta_min=1e-6)
model = model.to(device)
train_loss = []
test_loss = []
best_score = 100.0

for epoch in range(epochs):

    model.train()
    train_total_loss = 0.0
    test_total_loss = 0.0
    loss = 0
    for input_, target in train:
        input_ = input_.to(device)
        target = target.to(device)

        out = model(input_)

        loss = loss_fun(out, target)

        optimier.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimier.step()
        train_total_loss += loss.item()

    train_loss.append(train_total_loss / len(train))
    model.eval()
    with torch.no_grad():
        for input_, target in test:
            input_ = input_.to(device)
            target = target.to(device)

            out = model(input_)

            loss = loss_fun(out, target)

            test_total_loss += loss.item()

        test_loss.append(test_total_loss / len(test))
        scheduler.step()
        if best_score > test_loss[-1]:
            torch.save(model.state_dict(), "best_model.pth")
            best_score = test_loss[-1]
        current_lr = optimier.param_groups[0]["lr"]
        print(f"epoch:{epoch + 1}的train_loss,test_loss：{train_loss[epoch], test_loss[epoch]}，lr：{current_lr}")
plt.close('all')
plt.plot(train_loss, color='blue', label='Train Loss')
plt.plot(test_loss, color='red', label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(1, epochs)
plt.title("Training And Testing Loss Curve")
plt.show()