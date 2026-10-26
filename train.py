import sys, json, time, gc
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import tools as t

# 数据与路径配置
out_channals = 32
batch_size = 32
elec = 6
window_length = 10 * 256
window_overlap = 5 * 256
elec_names = ['Fp1-F7', 'Fp2-F8', 'T7-P7', 'T8-P8', 'P7-O1', 'P8-O2']
data_path = r"D:\data\dianxian\wang2021\chb22"

# 划分比例
encoder_val = 0.1
classfy_val = 0.1

# 模型与训练超参数
seed = 0
channal_attention = False
temporal_attention = True
hidden_size = 20
dropout = 0.7           # 防止过拟合
learning_rate = 1e-4    # 降低学习率
weight_decay = 1e-3     # L2正则化
patience = 10
epochs = 30


class ConvPaddingSame(nn.Module):
    def __init__(self):
        super().__init__()

    def Conv1d(self, in_c, out_c, kernel_size, stride=1, groups=1, bias=False):
        padding = kernel_size // 2
        return nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def ConvTranspose1d(self, in_c, out_c, kernel_size, stride=1, groups=1, bias=False):
        padding = kernel_size // 2
        output_padding = stride - 1
        return nn.ConvTranspose1d(in_c, out_c, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, output_padding=output_padding)

# 卷积头
class CommonConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.cps = ConvPaddingSame()
        self.cl = [1, 2, 4, 8, 16, 32]
        self.model = self.build_model()

    def cell_layer(self, in_c, out_c):
        return nn.Sequential(
            self.cps.Conv1d(in_c, out_c, kernel_size=3, stride=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def build_model(self,):
        m = [self.cell_layer(self.cl[i - 1], self.cl[i]) for i in range(1, len(self.cl))]
        return nn.Sequential(*m)

    def forward(self, x):
        return self.model(x)

# 时序注意力
class TemporalAttention(nn.Module):
    def __init__(self, channals):
        super().__init__()
        self.fc = nn.Linear(channals, 1)

    def attention(self, x):
        scores = self.fc(x)
        weights = F.softmax(scores, dim=1)
        atten = weights * x
        return weights, atten

    def forward(self, x):
        return self.attention(x)

# 通道注意力
class ChannalAttention(nn.Module):
    def __init__(self, channals):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channals, channals // 2),
            nn.LayerNorm(channals // 2),
            nn.ReLU(),
            nn.Linear(channals // 2, channals),
            nn.Sigmoid()
        )

    def attention(self, x):
        pooled = x.mean(dim=1)
        weights = self.fc(pooled).unsqueeze(dim=1)
        atten = weights * x
        return weights, atten

    def forward(self, x):
        return self.attention(x)

# 双向LSTM
class BILSTMClassify(nn.Module):
    def __init__(self, input_size=out_channals, hidden_size=10, num_layers=1, dropout=0, temporal_attention=False, channal_attention=False):
        super().__init__()
        self.BI_LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.temporal_attention = temporal_attention
        self.channal_attention = channal_attention
        if self.temporal_attention:
            self.temp_atten = TemporalAttention(2 * hidden_size)
        if self.channal_attention:
            self.chan_atten = ChannalAttention(2 * hidden_size)

    def forward(self, x):
        x, _ = self.BI_LSTM(x)
        if self.channal_attention:
            _, x = self.chan_atten(x)
        if self.temporal_attention:
            _, x = self.temp_atten(x)
            out = x.sum(dim=1)
        else:
            out = x[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# 分类器
class ConvBILSTMClassify(nn.Module):
    def __init__(self, input_size=out_channals, hidden_size=hidden_size, num_layers=1, dropout=0, temporal_attention=False, channal_attention=False):
        super().__init__()
        self.common_conv = CommonConv1d()
        self.classify = BILSTMClassify(out_channals, hidden_size, num_layers, dropout, temporal_attention, channal_attention)

    def forward(self, x):
        x = self.common_conv(x)
        x = x.permute(0, 2, 1)
        return self.classify(x)


def train_val_test(elec_names, meta, windows, seed, val=(encoder_val, classfy_val), method="LOOCV"):
    dataset = []
    for elec in elec_names:
        data_set = t.split_train_test(meta[elec], seed=seed, val=val, method=method)
        elec_list = []
        for fold in range(len(data_set)):
            encoder_train_idx = data_set[fold]["auto_encoder_idx"]
            encoder_val_idx = data_set[fold]["auto_encoder_val_idx"]
            train_idx = data_set[fold]["train_idx"]
            val_idx = data_set[fold]["val_idx"]
            test_idx = data_set[fold]["test_idx"]
            wins_and_label = {}

            wins_and_label["encoder_train_wins"] = torch.tensor(windows[elec][encoder_train_idx][:, None, :], dtype=torch.float32)
            wins_and_label["encoder_val_wins"] = torch.tensor(windows[elec][encoder_val_idx][:, None, :], dtype=torch.float32)
            wins_and_label["train_wins"] = torch.tensor(windows[elec][train_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_train"] = torch.tensor(meta[elec].iloc[train_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)
            wins_and_label["val_wins"] = torch.tensor(windows[elec][val_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_val"] = torch.tensor(meta[elec].iloc[val_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)
            wins_and_label["test_wins"] = torch.tensor(windows[elec][test_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_test"] = torch.tensor(meta[elec].iloc[test_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)

            elec_list.append(wins_and_label)

        dataset.append(elec_list)
    return dataset


def train_elec_model(model, optimizer, loss_function, dataset, elec_idx, device='cuda', patience=5, epochs=30):
    model.to(device)
    now = datetime.now().strftime("%y%m_%d_%H%M%S")
    fold_paths = []
    fold_logs = {}


    for fold in range(len(dataset[elec_idx])):
        train_dataloader = DataLoader(TensorDataset(dataset[elec_idx][fold]["train_wins"], dataset[elec_idx][fold]["y_train"]), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(dataset[elec_idx][fold]["val_wins"], dataset[elec_idx][fold]["y_val"]), batch_size=batch_size, shuffle=False)

        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None
        fold_logs[f"fold_{fold + 1}"] = {"epochs": []}

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            val_loss = 0

            model.train()
            for (x, y) in train_dataloader:
                x, y = x.to("cuda"), y.to("cuda")
                optimizer.zero_grad()
                out = model(x)
                loss = loss_function(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)

            model.eval()
            with torch.no_grad():
                val_targets = []
                val_probs = []
                for (x, y) in val_dataloader:
                    x, y = x.to("cuda"), y.to("cuda")
                    out = model(x)
                    loss = loss_function(out, y)
                    val_loss += loss.item()
                    probs = torch.sigmoid(out)
                    val_probs.extend(probs.cpu().numpy().flatten())
                    val_targets.extend(y.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_dataloader)
            val_auc = roc_auc_score(val_targets, val_probs)

            end_time = time.time()

            fold_logs[f"fold_{fold + 1}"]["epochs"].append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "val_auc": round(val_auc, 4),
                "time": round(end_time - start_time, 2)
            })

            print(f"Epoch [{epoch+1:02d}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.2e} | "
                  f"Val Loss: {avg_val_loss:.2e} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Time: {end_time - start_time:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Early Stopping")
                    fold_logs[f"fold_{fold + 1}"]["early_stop_epoch"] = epoch + 1
                    break

        if best_model_state is not None:
            path = root / "weight" / f"{now}_{elec_names[elec_idx]}_fold{fold+1}.pth"
            fold_paths.append(path)
            torch.save(best_model_state, path)
            fold_logs[f"fold_{fold + 1}"]["weight_path"] = path
            fold_logs[f"fold_{fold + 1}"]["best_val_loss"] = round(best_val_loss, 6)

    return fold_paths,fold_logs


if __name__ == "__main__":
    print("正在划分数据")
    windows, meta = t.preprocess(r"D:\data\dianxian\wang2021\chb22", SPH=2 * 60, SOP=30 * 60, interictal_thr=2 * 60 * 60, window_length=4, window_overlap=2, elec_names=elec_names)

    models = []
    for i in range(len(elec_names)):
        model = ConvBILSTMClassify(channal_attention=channal_attention, temporal_attention=temporal_attention,dropout=dropout)
        models.append(model)

    dataset = train_val_test(elec_names, meta, windows, seed=seed, val=(0.1, 0.1))

    now = datetime.now().strftime("%y%m_%d_%H%M%S")
    log_dir = root / "train_log"
    combined_manifest = {
        "config": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "dropout": dropout,
            "patience": patience,
            "epochs": epochs,
            "channal_attention": channal_attention,
            "temporal_attention": temporal_attention,
            "hidden_size": hidden_size
        },
        "weights": [],  # 全局只留一份权重列表
        "logs": {}  # 按电极名分别记录详细日志
    }
    for i, model in enumerate(models):
        print(f"正在训练model：{elec_names[i]}")
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        loss_function = nn.BCEWithLogitsLoss()
        fold_paths,fold_logs = train_elec_model(model, optimizer, loss_function, dataset, elec_idx=i, device='cuda', patience=patience, epochs=epochs)

        combined_manifest["weights"].extend(fold_paths)  # 把这一电极的所有权重路径拼进去
        combined_manifest["logs"][elec_names[i]] = fold_logs  # 按电极名存入日志

        del model, optimizer, loss_function
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[清理] {elec_names[i]} 训练完成，显存已释放。")

    json_path = log_dir / f"{now}_weights_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, indent=4, ensure_ascii=False, default=str)
