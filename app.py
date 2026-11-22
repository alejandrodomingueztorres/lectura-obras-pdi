# app.py: punto de entrada de la app (Hugging Face Spaces ejecuta este archivo)
import os
import pandas as pd
import torch
import lectura_obras as lo
import red_epoca as rn
import interfaz

torch.set_num_threads(2)
CLASES = ['1600-1799', '1800-1869', '1870-1913', '1914-1999']   # mismo orden que en el cuaderno (sorted)
UMBRAL_EPOCA = 0.5
AQUI = os.path.dirname(os.path.abspath(__file__))

# modelo de época: ResNet18 con el último bloque afinado (variante E del cuaderno)
red = rn.crear_red(len(CLASES), preentrenada=False, descongelar='layer4.1')
red.load_state_dict(torch.load(os.path.join(AQUI, 'modelos', 'red_epoca_layer4.1.pth'), map_location='cpu'))
red.eval()
predictor = lambda rgb: rn.predecir_epoca_red(red, rgb, CLASES, UMBRAL_EPOCA)

# modelo de expresión: FER+ (ONNX) y perfil de la colección para los percentiles
lector = lo.LectorRostro(os.path.join(AQUI, 'modelos', 'ferplus.onnx'), umbral=0.55)
perfil = pd.read_csv(os.path.join(AQUI, 'datos', 'perfil_coleccion.csv'))

demo = interfaz.crear_interfaz(lector, tabla_coleccion=perfil, predictor_epoca=predictor, umbral_epoca=UMBRAL_EPOCA)

if __name__ == '__main__':
    demo.launch()
