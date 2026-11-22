---
title: Lectura de obras de arte con PDI
emoji: 🎨
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# Lectura de obras de arte con procesamiento digital de imágenes

Proyecto final de Procesamiento Digital de Imágenes (UAO). Autores: Santiago López Gómez, Alejandro Domínguez Torres y Juan Jacob Hernández. Docente: Nicolas Llanos Neuta.

El estudiante carga una pintura y el sistema muestra:

- la **paleta** de colores y el mapa de **luz y sombra**;
- las **zonas de interés** (saliencia clásica) y una lectura de cada una: color, posición, brillo, calidez, saturación y contraste;
- opcionalmente, una **estimación de la época** (4 rangos de años) con su confianza, hecha con una ResNet18 con el último bloque afinado;
- opcionalmente, con consentimiento, la **expresión facial estimada** (FER+) cruzada con lo que el estudiante escribe.

## Límites que se declaran

- La expresión facial **no equivale a la emoción**: es una referencia visual, no un diagnóstico.
- Expresión (72 fotos, 3 personas): cuando concluye acierta 0.84, pero solo concluye en el 39 % de los casos; casi no reconoce tristeza ni enojo.
- Época: exactitud 0.667 en test (n = 57, IC95 % [0.54, 0.78]) frente a 0.25 de azar. Brecha train–test de 0.27, por encima del criterio propio de 0.25.
- Zonas: aciertan 0.27 de los clics de 4 observadores frente a 0.14 de azar, pero no superan a los cuatro cuadros fijos de la regla de tercios (0.34).
- Las fotos se procesan en memoria y no se guardan (Ley 1581 de 2012). Este Space se ejecuta en un servidor de Hugging Face: no suba fotos de terceros.

## Ejecutar en local

```bash
pip install -r requirements.txt gradio
python app.py
```
