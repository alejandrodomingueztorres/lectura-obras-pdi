# red_epoca.py
# Ajuste fino PARCIAL de una ResNet18 para estimar la epoca, pensado para pocas imagenes:
# casi toda la red queda congelada, solo se entrena el ultimo bloque residual y una cabeza chica, con aumentos,
# suavizado de etiquetas, decaimiento de pesos y parada temprana. Incluye Grad-CAM para ver donde mira la red.
import copy
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from PIL import Image

MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def cargar_en_memoria(rutas, lado=256):
    # una sola lectura de disco: las imagenes (lado corto = 'lado') quedan en RAM para todas las epocas
    out = []
    for r in rutas:
        im = Image.open(r).convert('RGB')
        w, h = im.size
        esc = lado / min(w, h)
        out.append(im.resize((max(int(w * esc), lado), max(int(h * esc), lado)), Image.BILINEAR))
    return out


class _Datos(torch.utils.data.Dataset):
    def __init__(self, imgs, y, tf):
        self.imgs, self.y, self.tf = imgs, y, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.y[i])


TF_TRAIN = T.Compose([T.RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),   # sin matiz: el color es parte de la epoca
                      T.ToTensor(), T.Normalize(MEDIA, DESV)])
TF_EVAL = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEDIA, DESV)])


def crear_red(n_clases, preentrenada=True, p_drop=0.3, descongelar='layer4.1'):
    # descongelar: 'cabeza' (solo la capa final), 'layer4.1' (ultimo bloque residual, ~4.7 M) o 'layer4' (~8.4 M)
    pesos = torchvision.models.ResNet18_Weights.DEFAULT if preentrenada else None
    red = torchvision.models.resnet18(weights=pesos)
    for p in red.parameters():
        p.requires_grad = False
    if descongelar != 'cabeza':
        bloque = red.layer4[1] if descongelar == 'layer4.1' else red.layer4
        for p in bloque.parameters():
            p.requires_grad = True
    red.fc = nn.Sequential(nn.Dropout(p_drop), nn.Linear(512, n_clases))
    return red


def _modo_entrenamiento(red):
    # BatchNorm siempre con las estadisticas de ImageNet: con 400 imagenes las estadisticas de un lote son demasiado ruidosas
    red.train()
    for m in red.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def afinar(imgs, y, idx_tr, idx_va, clases, epocas=15, paciencia=4, lr_cabeza=1e-3, lr_bloque=1e-4,
           decaimiento=1e-2, suavizado=0.1, lote=32, semilla=7, preentrenada=True, descongelar='layer4.1',
           dispositivo=None, verbose=True):
    # y: etiquetas texto; clases: lista ordenada. Devuelve (red con los mejores pesos de validacion, historial)
    dev = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(semilla); np.random.seed(semilla)
    ymap = {c: i for i, c in enumerate(clases)}
    yi = np.array([ymap[v] for v in y])
    tr = _Datos([imgs[i] for i in idx_tr], yi[idx_tr], TF_TRAIN)
    va = _Datos([imgs[i] for i in idx_va], yi[idx_va], TF_EVAL)
    g = torch.Generator().manual_seed(semilla)
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=lote, shuffle=True, generator=g, num_workers=0)
    dl_va = torch.utils.data.DataLoader(va, batch_size=64, shuffle=False, num_workers=0)

    red = crear_red(len(clases), preentrenada, descongelar=descongelar).to(dev)
    grupos = [{'params': red.fc.parameters(), 'lr': lr_cabeza}]
    libres = [p for n_, p in red.named_parameters() if p.requires_grad and not n_.startswith('fc.')]
    if libres:
        grupos.insert(0, {'params': libres, 'lr': lr_bloque})
    opt = torch.optim.AdamW(grupos, weight_decay=decaimiento)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    perdida = nn.CrossEntropyLoss(label_smoothing=suavizado)

    hist, mejor, mejor_pesos, sin_mejora = [], float('inf'), None, 0
    for ep in range(1, epocas + 1):
        _modo_entrenamiento(red)
        s_p, s_a, n = 0.0, 0, 0
        for x, t in dl_tr:
            x, t = x.to(dev), t.to(dev)
            sal = red(x)
            p = perdida(sal, t)
            opt.zero_grad(); p.backward(); opt.step()
            s_p += p.item() * len(x); s_a += (sal.argmax(1) == t).sum().item(); n += len(x)
        sched.step()
        red.eval()
        v_p, v_a, m = 0.0, 0, 0
        with torch.no_grad():
            for x, t in dl_va:
                x, t = x.to(dev), t.to(dev)
                sal = red(x)
                v_p += nn.functional.cross_entropy(sal, t, reduction='sum').item()
                v_a += (sal.argmax(1) == t).sum().item(); m += len(x)
        hist.append({'epoca': ep, 'perdida_train': s_p / n, 'acc_train': s_a / n, 'perdida_val': v_p / m, 'acc_val': v_a / m})
        marca = ''
        if v_p / m < mejor - 1e-4:
            mejor, mejor_pesos, sin_mejora, marca = v_p / m, copy.deepcopy(red.state_dict()), 0, ' *'
        else:
            sin_mejora += 1
        if verbose:
            print('época %2d | train: pérdida %.3f acc %.3f | val: pérdida %.3f acc %.3f%s' % (ep, s_p / n, s_a / n, v_p / m, v_a / m, marca))
        if sin_mejora >= paciencia:
            if verbose:
                print('parada temprana: la validación no mejora en %d épocas' % paciencia)
            break
    red.load_state_dict(mejor_pesos)
    return red.eval(), pd.DataFrame(hist)


def predecir_proba(red, imgs, lote=64, dispositivo=None):
    # promedia la imagen y su volteo horizontal (mismo aumento que usa la regresion logistica)
    dev = dispositivo or next(red.parameters()).device
    red.eval()
    salida = []
    with torch.no_grad():
        for i in range(0, len(imgs), lote):
            x = torch.stack([TF_EVAL(im) for im in imgs[i:i + lote]]).to(dev)
            p = (torch.softmax(red(x), 1) + torch.softmax(red(torch.flip(x, dims=[3])), 1)) / 2
            salida.append(p.cpu().numpy())
    return np.concatenate(salida)


def predecir_epoca_red(red, rgb, clases, umbral=0.5):
    # misma salida que modelo_epoca.predecir_epoca, para poder enchufarla en la interfaz
    pr = predecir_proba(red, [Image.fromarray(np.asarray(rgb, np.uint8))])[0]
    orden = np.argsort(-pr)
    top = [(str(clases[i]), float(pr[i])) for i in orden[:2]]
    if pr[orden[0]] < umbral:
        return {'estado': 'epoca_incierta', 'epoca': None, 'confianza': top[0][1], 'top2': top}
    return {'estado': 'ok', 'epoca': top[0][0], 'confianza': top[0][1], 'top2': top}


def gradcam(red, imagen, clase=None):
    # mapa de calor del ultimo bloque: que partes de la obra empujan la decision. Devuelve (mapa 0-1 del tamaño de la imagen, clase, prob)
    dev = next(red.parameters()).device
    if not isinstance(imagen, Image.Image):
        imagen = Image.fromarray(np.asarray(imagen, np.uint8))
    x = TF_EVAL(imagen).unsqueeze(0).to(dev)
    guardado = {}
    h1 = red.layer4.register_forward_hook(lambda m, i, o: guardado.__setitem__('a', o))
    h2 = red.layer4.register_full_backward_hook(lambda m, gi, go: guardado.__setitem__('g', go[0]))
    try:
        red.eval()
        with torch.enable_grad():
            x.requires_grad_(True)
            sal = red(x)
            prob = torch.softmax(sal, 1)[0].detach()
            c = int(sal.argmax(1)) if clase is None else int(clase)
            red.zero_grad()
            sal[0, c].backward()
    finally:
        h1.remove(); h2.remove()
    alfa = guardado['g'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((alfa * guardado['a']).sum(1))[0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)
    w, h = imagen.size
    return cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1), c, float(prob[c])