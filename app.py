# app.py: punto de entrada de la app. Usa solo ONNX Runtime (sin PyTorch), por eso cabe en servicios gratuitos pequeños.
import os
import pandas as pd
import lectura_obras as lo
import onnx_epoca
import interfaz

CLASES = ['1600-1799', '1800-1869', '1870-1913', '1914-1999']   # mismo orden que en el cuaderno (sorted)
UMBRAL_EPOCA = 0.5
AQUI = os.path.dirname(os.path.abspath(__file__))

predictor = onnx_epoca.RedEpoca(os.path.join(AQUI, 'modelos', 'red_epoca.onnx'), CLASES, UMBRAL_EPOCA)
lector = lo.LectorRostro(os.path.join(AQUI, 'modelos', 'ferplus.onnx'), umbral=0.55)
perfil = pd.read_csv(os.path.join(AQUI, 'datos', 'perfil_coleccion.csv'))

demo = interfaz.crear_interfaz(lector, tabla_coleccion=perfil, predictor_epoca=predictor, umbral_epoca=UMBRAL_EPOCA)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))