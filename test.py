dicts = torch.load("best_model.pth")
model.load_state_dict(dicts)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

def weighted_acc_simple(
    pred,
    target
):
    pred = pred.reshape(pred.shape[0],-1)
    target = target.reshape(target.shape[0],-1)

    pred = pred - pred.mean(dim=1,keepdim=True)
    target = target - target.mean(dim=1,keepdim=True)

    numerator = (pred*target).sum(dim=1)

    denominator = torch.sqrt(
        (pred**2).sum(dim=1)
        *
        (target**2).sum(dim=1)
    )

    acc = numerator/(denominator+1e-8)

    return acc.mean()

model.eval()


surface_vars = {
    "U10":0,
    "V10":1,
    "T2M":2,
    "MSL":3,
    "U1000":4,
    "U850":5,
    "U500":6,
    "U250":7,
    "V1000":8,
    "V850":9,
    "V500":10,
    "V250":11,
    "Z1000":12,
    "Z850":13,
    "Z500":14,
    "Z250":15,
    "T850":16,
    "Q1000":17,
    "Q850":18,
    "Q500":19,
}
metrics = {}

for name in list(surface_vars.keys()):

    metrics[name] = {
        "rmse": [],
        "mae": [],
        "acc": []
    }

with torch.no_grad():

    for lead in range(1,2):

        print(f"\nEvaluating {lead*6}h")

        dataset = Loader(
            test_data,lead
        )

        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=48,
            shuffle=False,
            drop_last=True
        )

        rmse_sum = {
            name:0.0
            for name in metrics
        }

        mae_sum = {
            name:0.0
            for name in metrics
        }

        acc_sum = {
            name:0.0
            for name in metrics
        }

        for input_,target in test_loader:

            input_ = input_.to(device)

            target = target.to(device)

            preds = input_
            # autoregressive rollout
            for _ in range(lead):

                preds= model( preds )
            # ------------------
            # 反归一化
            # ------------------


            preds = (
                preds.to("cpu") * std + mean
            )

            target = (
                target.to("cpu")  * std + mean
            )

            # ======================
            # Surface
            # ======================

            for name,var_idx in surface_vars.items():

                pred = preds[:,var_idx]
                truth = target[:,var_idx]

                rmse = torch.sqrt(
                    torch.mean(
                        (pred-truth)**2
                    )
                )

                mae = torch.mean(
                    torch.abs(
                        pred-truth
                    )
                )
                # print(pred.shape,truth.shape)
                acc = weighted_acc_simple(
                    pred,
                    truth
                )

                rmse_sum[name] += rmse.item()
                mae_sum[name] += mae.item()
                acc_sum[name] += acc.item()

        # 保存当前lead结果

        for name in metrics:

            metrics[name]["rmse"].append(
                rmse_sum[name] / len(test_loader)
            )

            metrics[name]["mae"].append(
                mae_sum[name] / len(test_loader)
            )

            metrics[name]["acc"].append(
                acc_sum[name] / len(test_loader)
            )

import matplotlib.pyplot as plt
import numpy as np

up_vars = [
    "U10",
    "V10",
    "T2M",
    "MSL",
    "U1000",
    "U850",
    "U500",
    "U250",
    "V1000",
    "V850",
    "V500",
    "V250",
    "Z1000",
    "Z850",
    "Z500",
    "Z250",
    "T850",
    "Q1000",
    "Q850",
    "Q500"]

sample_idx = 5

fig, axes = plt.subplots(len(up_vars),3, figsize=(120,180))

for i,var in enumerate(up_vars):

    truth = target[sample_idx,i].cpu().numpy()

    pred = preds[sample_idx,i].detach().cpu().numpy()

    error = pred - truth

    vmax = max(np.max(truth),np.max(pred))

    vmin = min(np.min(truth), np.min(pred))

    axes[i,0].imshow(truth,cmap="jet", vmin=vmin, vmax=vmax )
    axes[i,0].set_title( f"{var} Truth" )

    axes[i,1].imshow( pred, cmap="jet", vmin=vmin, vmax=vmax)
    axes[i,1].set_title(f"{var} Prediction")

    axes[i,2].imshow(error,cmap="RdBu_r")
    axes[i,2].set_title(f"{var} Error")

plt.tight_layout()
plt.show()

lead_time = np.arange(1,20) * 6

plt.figure(figsize=(10,6))




for var in [
   "U10",
    "V10",
    "T2M",
    "MSL",
    "U1000",
    "U850",
    "U500",
    "U250",
    "V1000",
    "V850",
    "V500",
    "V250",
    "Z1000",
    "Z850",
    "Z500",
    "Z250",
    "T850",
    "Q1000",
    "Q850",
    "Q500"
]:

    plt.plot(
        lead_time,
        metrics[var]["acc"],
        label=var
    )

plt.xlabel("Lead Time (hours)")
plt.ylabel("ACC")

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.grid()

plt.show()