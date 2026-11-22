# onnx_epoca.py: la red de época (ResNet18 con el último bloque afinado) exportada a ONNX, sin PyTorch.
# Hace lo mismo que red_epoca.predecir_epoca_red: redimensiona a 224x224, normaliza, promedia la imagen y su volteo.
import numpy as np
import onnxruntime as ort
from PIL import Image

MEDIA = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
DESV = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


class RedEpoca:
    def __init__(self, ruta_onnx, clases, umbral=0.5):
        op = ort.SessionOptions()
        op.intra_op_num_threads = 2
        self.sesion = ort.InferenceSession(ruta_onnx, op, providers=['CPUExecutionProvider'])
        self.clases = list(clases)
        self.umbral = umbral

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def proba(self, rgb):
        im = Image.fromarray(np.asarray(rgb, np.uint8)).convert('RGB').resize((224, 224), Image.BILINEAR)
        x = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
        x = (x - MEDIA) / DESV
        p = (self._softmax(self.sesion.run(None, {'x': x})[0]) +
             self._softmax(self.sesion.run(None, {'x': x[:, :, :, ::-1].copy()})[0])) / 2
        return p[0]

    def __call__(self, rgb):
        pr = self.proba(rgb)
        orden = np.argsort(-pr)
        top = [(str(self.clases[i]), float(pr[i])) for i in orden[:2]]
        if pr[orden[0]] < self.umbral:
            return {'estado': 'epoca_incierta', 'epoca': None, 'confianza': top[0][1], 'top2': top}
        return {'estado': 'ok', 'epoca': top[0][0], 'confianza': top[0][1], 'top2': top}