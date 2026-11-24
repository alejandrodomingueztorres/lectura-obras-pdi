# Lectura de obras de arte con procesamiento digital de imágenes

Proyecto final de Procesamiento Digital de Imágenes, Universidad Autónoma de Occidente.
Autores: Santiago López Gómez, Alejandro Domínguez Torres y Juan Jacob Hernández. Docente: Nicolas Llanos Neuta.

- App en línea: https://lectura-obras-pdi.onrender.com (plan gratuito: si estuvo inactiva, tarda cerca de un minuto en despertar)
- Cuaderno de Colab: https://colab.research.google.com/drive/16if4Tp51N_pPcI3AhilZ3oSQWttUUz8q
- Prototipo en Figma: https://www.figma.com/design/XwGSFZOUNtyrY1EG8Ma9wt

## ¿Qué hace?

El estudiante carga una pintura y el sistema muestra:

- la paleta de colores y el mapa de luz y sombra;
- las zonas de interés (saliencia clásica) y una lectura de cada una: color, posición, brillo, calidez, saturación y contraste;
- opcionalmente, una estimación de la época (4 rangos de años) con su confianza, con una ResNet18 con el último bloque afinado;
- opcionalmente, con consentimiento, la expresión facial estimada (FER+), que se cruza con lo que el estudiante escribe.

La expresión facial no equivale a la emoción: es una referencia visual, no un diagnóstico.

## Guía de uso de la interfaz

La interfaz (Gradio) tiene cuatro pestañas:

1. **Carga y detección.** Cargue una pintura (JPG o PNG) en «Pintura», elija cuántos colores quiere en la paleta (3 a 8) y pulse «Analizar obra». Aparece la obra con las zonas de interés numeradas y una lectura de cada una. Si la imagen no se puede abrir o es muy pequeña, se muestra un aviso.
2. **Exploración de zonas.** Elija una zona y un lente (Original + zona, Paleta reducida o Luz y sombra), ajuste la opacidad y pulse «Ver zona».
3. **Registro.** Escriba qué siente al ver la zona. Si lo desea, marque el consentimiento y suba o tome una foto de su rostro (opcional) y pulse «Registrar». El sistema estima la expresión y la compara con lo escrito. Si no detecta rostro o no concluye, lo dice y no inventa un resultado.
4. **Síntesis.** Pulse «Ver síntesis» para ver el resumen de la sesión y descargar el registro en JSON.

La foto se procesa en memoria, con consentimiento, y no se guarda (Ley 1581 de 2012). No suba fotos de terceros.

## Instrucciones de ejecución

### Opción A. En línea
Abra https://lectura-obras-pdi.onrender.com.

### Opción B. En su computador (Python 3.11)
```bash
git clone https://github.com/TU_USUARIO/lectura-obras-pdi.git
cd lectura-obras-pdi
git lfs pull
pip install -r requirements.txt
python app.py
```
Abra http://127.0.0.1:7860 en el navegador. Los modelos (`modelos/red_epoca.onnx` y `modelos/ferplus.onnx`) están en el repositorio con Git LFS.

### Opción C. Con Docker
```bash
docker build -t lectura-obras .
docker run -p 7860:7860 lectura-obras
```

### Opción D. Despliegue en Render
Servicio Web con Python 3, Build Command `pip install -r requirements.txt`, Start Command `python app.py` y la variable `PYTHON_VERSION=3.11.9`.

## Estructura

```
app.py              punto de entrada
interfaz.py         interfaz de Gradio
lectura_obras.py    paleta, luz y sombra, saliencia, lectura de zonas, expresión
onnx_epoca.py       red de época en ONNX
modelo_epoca.py     modelo base de época (usado en el cuaderno)
red_epoca.py        entrenamiento y Grad-CAM de la red (usado en el cuaderno)
modelos/            red_epoca.onnx y ferplus.onnx
datos/              perfil_coleccion.csv
```

## Datos

611 imágenes de pinturas en 4 rangos de años (1600-1799, 1800-1869, 1870-1913, 1914-1999). Se eliminaron 37 repetidas (MD5 y dHash) y se usaron 574, con partición fija 80, 10, 10. Las imágenes no se incluyen en el repositorio por derechos de autor; se incluye `datos/perfil_coleccion.csv` con las estadísticas por obra para calcular percentiles. El proceso completo (datos, entrenamiento y validaciones) está en el cuaderno de Colab.

## Resultados

| Componente | Resultado |
|---|---|
| Época, modelo base | 0.526 de exactitud en prueba |
| Época, red afinada (último bloque) | 0.667 en prueba (IC 95 %: 0.54 a 0.78); azar 0.25 |
| Época con umbral de confianza 0.6 | responde el 70 % de las obras, con 0.80 de exactitud |
| Zonas frente a clics humanos (150 clics, 10 obras) | 0.27 de aciertos; azar 0.14; tercios fijos 0.34 |
| Estabilidad de zonas ante 7 alteraciones | 0.87 |
| Expresión (72 fotos, 3 personas) | acierta 0.84 cuando concluye, pero concluye en el 39 %; tristeza y enojo casi no se reconocen (0.06) |
| Matriz de verificación | 17 de 18 criterios cumplidos |

## Límites

- La brecha entre entrenamiento y prueba de la red de época es de 0.27, por encima del criterio propio de 0.25 (R14 no se cumple). Se probó una variante regularizada y no mejoró.
- No se detectó rostro en el 32 % de las fotos de prueba de expresión.
- Las zonas superan al azar, pero no a los cuatro cuadros fijos de la regla de tercios.
- La época es una estimación, no un dato histórico.
- Los criterios de la matriz de verificación son propios del equipo.