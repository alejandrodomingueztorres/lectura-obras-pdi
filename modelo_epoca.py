# modelo_epoca.py
# Estimacion de epoca (bloque 5, opcional). Red preentrenada solo como extractor + clasificador chico.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class Extractor:
    # resnet18 preentrenada en ImageNet, sin la capa final: solo saca un vector de 512 por imagen
    def __init__(self, lado=224):
        import torch, torchvision
        from torchvision import transforms as T
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        red = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        red.fc = torch.nn.Identity()
        self.red = red.eval().to(self.dev)
        self.tf = T.Compose([T.Resize((lado, lado)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _vectores(self, imagenes):
        x = self.torch.stack([self.tf(i) for i in imagenes]).to(self.dev)
        with self.torch.no_grad():
            return self.red(x).cpu().numpy(), self.red(self.torch.flip(x, dims=[3])).cpu().numpy()

    def lote(self, rutas, tam=32):
        from PIL import Image
        a, b = [], []
        for i in range(0, len(rutas), tam):
            v, ve = self._vectores([Image.open(r).convert('RGB') for r in rutas[i:i + tam]])
            a.append(v)
            b.append(ve)
        return np.concatenate(a), np.concatenate(b)

    def uno(self, img_rgb):
        from PIL import Image
        return self._vectores([Image.fromarray(img_rgb)])[0][0]


def armar_modelo(C=0.05, comp=64, n_clasicos=0, d_emb=512):
    # d_emb = 512 numeros de la red; n_clasicos = rasgos de PDI (color, luz, textura) pegados a la derecha.
    # La red se comprime con PCA y C chico penaliza fuerte: asi el modelo no puede ajustarse al detalle de cada imagen.
    lr = LogisticRegression(C=C, max_iter=3000)
    if n_clasicos == 0:
        return make_pipeline(StandardScaler(), PCA(n_components=comp, random_state=0), lr)
    if d_emb == 0:
        return make_pipeline(StandardScaler(), lr)
    from sklearn.compose import ColumnTransformer
    union = ColumnTransformer([('red', make_pipeline(StandardScaler(), PCA(n_components=comp, random_state=0)), slice(0, d_emb)),
                               ('pdi', StandardScaler(), slice(d_emb, d_emb + n_clasicos))])
    return make_pipeline(union, lr)


def rasgos_clasicos(img_rgb):
    # 38 numeros hechos solo con PDI clasico: color (Lab, HSV, matiz), borde y textura (LBP). Sin entrenar nada.
    import cv2
    from skimage.feature import local_binary_pattern
    from scipy.stats import skew
    img = np.asarray(img_rgb, np.uint8)
    h, w = img.shape[:2]
    esc = 256 / max(h, w)
    if esc < 1:
        img = cv2.resize(img, (max(int(w * esc), 8), max(int(h * esc), 8)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    L, a, b = lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()
    S, V = hsv[..., 1].ravel() / 255.0, hsv[..., 2].ravel() / 255.0
    f = [L.mean(), L.std(), float(skew(L)) if L.std() > 1e-6 else 0.0, np.percentile(L, 5), np.percentile(L, 95),
         a.mean(), a.std(), b.mean(), b.std(), S.mean(), S.std(), V.mean(), V.std()]
    borde = cv2.Canny(gris, 80, 160).mean() / 255.0
    gx, gy = cv2.Sobel(gris, cv2.CV_32F, 1, 0), cv2.Sobel(gris, cv2.CV_32F, 0, 1)
    f += [borde, float(np.sqrt(gx ** 2 + gy ** 2).mean()), float(np.log1p(cv2.Laplacian(gris, cv2.CV_32F).var()))]
    lbp = local_binary_pattern(gris, 8, 1, 'uniform')
    f += list(np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / lbp.size)
    matiz = hsv[..., 0].ravel() * 2.0
    f += list(np.histogram(matiz, bins=12, range=(0, 360), weights=S)[0] / max(S.sum(), 1e-9))
    return np.asarray(f, np.float32)


def lote_clasicos(rutas):
    from PIL import Image
    return np.stack([rasgos_clasicos(np.array(Image.open(r).convert('RGB'))) for r in rutas])


def validacion_cruzada(X, Xesp, y, C=0.05, folds=5, semilla=7, **cfg):
    # los vectores volteados solo entran al entrenamiento, nunca a la evaluacion
    y = np.asarray(y)
    cv = StratifiedKFold(folds, shuffle=True, random_state=semilla)
    accs, f1s = [], []
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        m = armar_modelo(C, **cfg).fit(np.vstack([X[tr], Xesp[tr]]), np.concatenate([y[tr], y[tr]]))
        pred[te] = m.predict(X[te])
        accs.append(accuracy_score(y[te], pred[te]))
        f1s.append(f1_score(y[te], pred[te], average='macro'))
    return {'C': C, 'acc_media': float(np.mean(accs)), 'acc_desv': float(np.std(accs)),
            'f1_media': float(np.mean(f1s)), 'por_pliegue': [round(a, 3) for a in accs],
            'matriz': confusion_matrix(y, pred)}


def elegir_C(X, Xesp, y, opciones=(0.005, 0.02, 0.05, 0.2, 1.0), **cfg):
    tabla = [validacion_cruzada(X, Xesp, y, C=c, **cfg) for c in opciones]
    mejor = max(tabla, key=lambda t: t['acc_media'])
    return mejor['C'], pd.DataFrame([{k: v for k, v in t.items() if k not in ('matriz', 'por_pliegue')} for t in tabla])


def wilson(aciertos, n, z=1.96):
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centro - mitad), float(centro + mitad)


def evaluar_proba(y, proba, clases):
    # exactitud, F1 macro e intervalo de Wilson a partir de probabilidades (sirve para cualquier modelo)
    y = np.asarray(y)
    pred = np.asarray(clases)[np.asarray(proba).argmax(axis=1)]
    acc = accuracy_score(y, pred)
    return {'n': int(len(y)), 'acc': float(acc), 'f1_macro': float(f1_score(y, pred, average='macro')),
            'ic95': wilson(int(round(acc * len(y))), len(y)), 'matriz': confusion_matrix(y, pred, labels=clases)}


def entrenar_y_evaluar(df, X, Xesp, C, **cfg):
    # entrena con train, mira val para decidir, y test se mira una sola vez
    y = df['clase'].to_numpy()
    tr = (df['conjunto'] == 'train').to_numpy()
    m = armar_modelo(C, **cfg).fit(np.vstack([X[tr], Xesp[tr]]), np.concatenate([y[tr], y[tr]]))
    res = {}
    for nombre in ('train', 'val', 'test'):
        s = (df['conjunto'] == nombre).to_numpy()
        res[nombre] = evaluar_proba(y[s], m.predict_proba(X[s]), m.classes_)
    return m, res


def curva_cobertura_proba(pr, y, clases, umbrales=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8)):
    # si el modelo solo responde cuando esta seguro, cuanto mejora la exactitud y cuantas obras quedan sin respuesta
    conf = pr.max(axis=1)
    pred = np.asarray(clases)[pr.argmax(axis=1)]
    filas = []
    for u in umbrales:
        responde = conf >= u
        acc = accuracy_score(np.asarray(y)[responde], pred[responde]) if responde.any() else float('nan')
        filas.append({'umbral': u, 'cobertura': float(responde.mean()), 'exactitud': float(acc), 'n_responde': int(responde.sum())})
    return pd.DataFrame(filas)


def curva_cobertura(modelo, X, y, umbrales=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8)):
    return curva_cobertura_proba(modelo.predict_proba(X), y, modelo.classes_, umbrales)


def predecir_epoca(modelo, emb, umbral=0.5):
    pr = modelo.predict_proba(emb.reshape(1, -1))[0]
    orden = np.argsort(-pr)
    top = [(str(modelo.classes_[i]), float(pr[i])) for i in orden[:2]]
    if pr[orden[0]] < umbral:
        return {'estado': 'epoca_incierta', 'epoca': None, 'confianza': top[0][1], 'top2': top}
    return {'estado': 'ok', 'epoca': top[0][0], 'confianza': top[0][1], 'top2': top}


def similares(emb, X_ref, rutas_ref, k=3):
    a = emb / (np.linalg.norm(emb) + 1e-9)
    b = X_ref / (np.linalg.norm(X_ref, axis=1, keepdims=True) + 1e-9)
    s = b @ a
    idx = np.argsort(-s)[:k]
    return [{'ruta': rutas_ref[i], 'similitud': float(s[i])} for i in idx]