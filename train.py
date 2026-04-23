import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PL(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out, inp))
        self.b = nn.Parameter(torch.zeros(out))
        self.g_param = nn.Parameter(torch.zeros(out, inp))
        nn.init.kaiming_uniform_(self.w, a=0.01)

    def forward(self, x):
        g = torch.sigmoid(self.g_param)
        w_eff = self.w * g
        return F.linear(x, w_eff, self.b)

    def gates(self):
        return torch.sigmoid(self.g_param).detach().cpu()

    def sp(self, t=1e-2):
        g = self.gates()
        return (g < t).float().mean().item()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = PL(3072, 512)
        self.b = PL(512, 256)
        self.c = PL(256, 128)
        self.d = PL(128, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.a(x))
        x = self.drop(x)
        x = F.relu(self.b(x))
        x = self.drop(x)
        x = F.relu(self.c(x))
        x = self.d(x)
        return x

    def layers(self):
        for m in self.modules():
            if isinstance(m, PL):
                yield m

    def sp_loss(self):
        s = torch.tensor(0.0)
        for l in self.layers():
            g = torch.sigmoid(l.g_param)
            s = s + g.abs().sum()
        return s

    def total_sp(self, t=1e-2):
        tot, pr = 0, 0
        for l in self.layers():
            g = l.gates()
            pr += (g < t).sum().item()
            tot += g.numel()
        return pr / tot if tot > 0 else 0.0

    def all_g(self):
        vals = []
        for l in self.layers():
            vals.append(l.gates().flatten().numpy())
        return np.concatenate(vals)

#might tune this later
def get_data(bs=128):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=tr)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=te)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch(model, loader, opt, lam, device):
    model.train()
    tot_l, cls_l, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss_cls = F.cross_entropy(out, y)
        loss_sp = model.sp_loss().to(device)

        loss = loss_cls + lam * loss_sp
        loss.backward()
        opt.step()

        tot_l += loss.item()
        cls_l += loss_cls.item()
        n += 1

    return tot_l / n, cls_l / n


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    c, t = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        c += (pred == y).sum().item()
        t += y.size(0)

    return c / t


def run(lam, ep, device, tr_loader, te_loader):
    print("\nRunning with lambda:", lam)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)

    hist = {"loss": [], "cls": [], "acc": [], "sp": []}

    for e in range(1, ep + 1):
        tl, cl = train_epoch(model, tr_loader, opt, lam, device)
        acc = test(model, te_loader, device)
        sp = model.total_sp()
        sch.step()

        hist["loss"].append(tl)
        hist["cls"].append(cl)
        hist["acc"].append(acc)
        hist["sp"].append(sp)

        if e % 5 == 0 or e == 1:
            print(e, tl, cl, acc, sp)

    g_vals = model.all_g()

    return {
        "lam": lam,
        "acc": acc,
        "sp": sp,
        "g": g_vals,
        "hist": hist
    }


def plot_res(res, idx):
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    best = res[idx]

    ax1.hist(best["g"], bins=80)
    ax1.set_title("Gate Distribution")

    ax2 = fig.add_subplot(gs[1])
    sps = [r["sp"] * 100 for r in res]
    accs = [r["acc"] * 100 for r in res]

    ax2.scatter(sps, accs)
    ax2.set_xlabel("Sparsity")
    ax2.set_ylabel("Accuracy")

    plt.savefig("gate_distribution.png")
    plt.show()

#basic training loops
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader, te_loader = get_data(256)

    lams = [1e-5, 1e-4, 5e-4]
    epochs = 30

    results = []

    for lam in lams:
        r = run(lam, epochs, device, tr_loader, te_loader)
        results.append(r)

    for r in results:
        print(r["lam"], r["acc"], r["sp"])

    best_i = max(range(len(results)), key=lambda i: results[i]["acc"])
    plot_res(results, best_i)


if __name__ == "__main__":
    main()
