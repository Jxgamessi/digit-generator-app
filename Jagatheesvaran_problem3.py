import streamlit as s
import torch
import torch.nn as n
import torch.nn.functional as f
import matplotlib.pyplot as p

class G(n.Module):
    def __init__(z):
        super().__init__()
        z.e = n.Embedding(10, 10)
        z.a = n.Linear(784 + 10, 256)
        z.b = n.Linear(256, 512)
        z.c = n.Linear(512, 784)

    def forward(z, i, y):
        y = z.e(y)
        i = i.view(i.size(0), -1)
        i = torch.cat([i, y], dim=1)
        i = f.relu(z.a(i))
        i = f.relu(z.b(i))
        i = torch.sigmoid(z.c(i))
        return i.view(-1, 1, 28, 28)

m = G()
m.load_state_dict(torch.load("m.pth", map_location=torch.device('cpu')))
m.eval()

s.title("Digit Gen 🧠")
v = s.selectbox("Pick digit:", list(range(10)))

if s.button("Generate"):
    y = torch.tensor([v]*5)
    i = torch.rand(5, 1, 28, 28)
    with torch.no_grad():
        o = m(i, y)
    for a in o:
        p.imshow(a.squeeze(), cmap='gray')
        s.pyplot(p)
