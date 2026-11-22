# lectura_obras.py
# Bloques del pipeline: preproceso, lentes, zona foco, lectura de la zona, rostro y expresion, cruce.
import os, re, unicodedata
import numpy as np
import cv2
from scipy.stats import skew
from skimage.filters import threshold_multiotsu
from skimage.feature import peak_local_max

LADO = 512
EXPRESIONES_FERPLUS = ['neutral', 'felicidad', 'sorpresa', 'tristeza', 'enojo', 'asco', 'miedo', 'desprecio']


# ---------- 1 y 2: entrada y preproceso ----------

def abrir_imagen(origen, lado=LADO):
    # origen: ruta, bytes o arreglo RGB
    estado = 'ok'
    try:
        if isinstance(origen, str):
            bgr = cv2.imread(origen, cv2.IMREAD_COLOR)
            img = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(origen, (bytes, bytearray)):
            bgr = cv2.imdecode(np.frombuffer(origen, np.uint8), cv2.IMREAD_COLOR)
            img = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            img = np.asarray(origen)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img[:, :, :3].astype(np.uint8)
    except Exception:
        img = None
    if img is None or img.size == 0:
        return None, 'imagen_invalida'
    h, w = img.shape[:2]
    if min(h, w) < 64:
        return None, 'imagen_muy_pequena'
    esc = lado / max(h, w)
    if esc < 1:
        img = cv2.resize(img, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
    return img, estado


def preparar(img):
    f = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)          # L 0-100, a y b aprox -128..127
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {'rgb': img, 'lab': lab, 'hsv': hsv, 'gris': gris}


# ---------- 3: lentes de color y luz ----------

def paleta(p, k=5, semilla=3):
    lab = p['lab']
    h, w = lab.shape[:2]
    datos = lab.reshape(-1, 3)
    rng = np.random.RandomState(semilla)
    muestra = datos[rng.choice(len(datos), min(20000, len(datos)), replace=False)]
    cv2.setRNGSeed(semilla)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centros = cv2.kmeans(muestra, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = ((datos[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    et = d.argmin(axis=1)
    pct = np.bincount(et, minlength=k) / len(et)
    orden = np.argsort(-pct)
    centros, pct = centros[orden], pct[orden]
    rgb_c = cv2.cvtColor(centros.reshape(1, k, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(k, 3)
    rgb_c = np.clip(rgb_c * 255, 0, 255).astype(np.uint8)
    nuevo = np.zeros(len(et), dtype=int)
    for pos, viejo in enumerate(orden):
        nuevo[et == viejo] = pos
    vista = rgb_c[nuevo].reshape(h, w, 3)
    return {'colores': rgb_c, 'porcentajes': pct, 'vista': vista}


def luz_sombra(p, clases=3):
    L = p['lab'][:, :, 0]
    if L.max() - L.min() < 5:
        return {'estado': 'imagen_plana'}
    try:
        umbrales = threshold_multiotsu(L, classes=clases)
    except ValueError:
        return {'estado': 'imagen_plana'}
    mapa = np.digitize(L, umbrales)                   # 0 sombra, 1 medios, 2 luces
    pct = np.bincount(mapa.ravel(), minlength=clases) / mapa.size
    hist, _ = np.histogram(L, bins=32, range=(0, 100))
    gris = np.array([40, 135, 225], dtype=np.uint8)[mapa]
    return {'estado': 'ok', 'umbrales': [float(u) for u in umbrales], 'mapa': mapa,
            'sombra': float(pct[0]), 'medios': float(pct[1]), 'luces': float(pct[2]),
            'brillo_medio': float(L.mean()), 'contraste': float(L.std()),
            'asimetria': float(skew(L.ravel())), 'histograma': hist,
            'vista': np.dstack([gris, gris, gris])}


def mezclar(original, vista, alfa=0.6):
    # alfa = cuanto se ve la vista transformada
    return cv2.addWeighted(vista, alfa, original, 1 - alfa, 0)


# ---------- 4: zona foco ----------

def saliencia(p):
    # saliencia por frecuencia sintonizada: distancia en Lab entre el color medio y la imagen suavizada
    lab = cv2.GaussianBlur(p['lab'], (0, 0), 3)
    medio = p['lab'].reshape(-1, 3).mean(axis=0)
    s = ((lab - medio) ** 2).sum(axis=2)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    h, w = s.shape
    ys, xs = np.mgrid[0:h, 0:w]
    sig = 0.15 * min(h, w)
    cerca = np.zeros_like(s)
    for fx in (1 / 3, 2 / 3):
        for fy in (1 / 3, 2 / 3):
            cerca = np.maximum(cerca, np.exp(-(((xs - fx * w) ** 2 + (ys - fy * h) ** 2) / (2 * sig ** 2))))
    s2 = s * (0.8 + 0.2 * cerca)                      # pequeño empuje hacia los puntos de tercios
    return cv2.GaussianBlur(s2, (0, 0), 6), cerca


def puntos_interes(mapa, cerca, n=4):
    h, w = mapa.shape
    lado = int(0.24 * min(h, w))
    picos = peak_local_max(mapa, min_distance=lado, threshold_rel=0.25, num_peaks=n, exclude_border=False)
    puntos = []
    for (y, x) in picos:
        x0 = int(np.clip(x - lado // 2, 0, w - lado))
        y0 = int(np.clip(y - lado // 2, 0, h - lado))
        peso = float(mapa[y0:y0 + lado, x0:x0 + lado].mean())
        col = min(int(3 * x / w), 2)
        fila = min(int(3 * y / h), 2)
        puntos.append({'x': x0, 'y': y0, 'w': lado, 'h': lado, 'peso': peso,
                       'celda': ['arriba', 'centro', 'abajo'][fila] + '-' + ['izquierda', 'centro', 'derecha'][col],
                       'cerca_tercios': float(cerca[y, x])})
    puntos.sort(key=lambda q: -q['peso'])
    for i, q in enumerate(puntos, 1):
        q['id'] = i
    return puntos


def marcar_puntos(rgb, puntos, elegido=None):
    salida = rgb.copy()
    for q in puntos:
        color = (255, 60, 60) if q['id'] == elegido else (255, 255, 255)
        cv2.rectangle(salida, (q['x'], q['y']), (q['x'] + q['w'], q['y'] + q['h']), color, 2)
        cv2.putText(salida, str(q['id']), (q['x'] + 6, q['y'] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return salida


# ---------- 6: lectura de la zona ----------

def nombre_color(rgb):
    # nombre aproximado en español a partir de matiz, saturacion y valor (reglas simples, pensadas para pintura)
    r, g, b = [int(v) for v in rgb]
    h, s, v = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = h * 2.0, s / 255.0, v / 255.0
    if v < 0.15:
        return 'negro'
    if s < 0.12:
        return 'blanco' if v > 0.85 else ('gris claro' if v > 0.55 else 'gris oscuro')
    if h < 12 or h >= 345:
        return 'granate' if v < 0.45 else 'rojo'
    if h < 45:
        if v < 0.55:
            return 'marrón'
        return 'beige' if s < 0.45 else 'naranja'
    if h < 70:
        return 'ocre' if v < 0.7 or s < 0.6 else 'amarillo'
    if h < 160:
        return 'verde oliva' if v < 0.45 else 'verde'
    if h < 200:
        return 'turquesa'
    if h < 260:
        return 'azul oscuro' if v < 0.4 else 'azul'
    if h < 300:
        return 'violeta'
    return 'rosa'


def frase_posicion(celda):
    fila, col = celda.split('-')
    if fila == 'centro' and col == 'centro':
        return 'en el centro'
    if fila == 'centro':
        return 'a la %s' % col
    if col == 'centro':
        return fila
    return '%s a la %s' % (fila, col)


def leer_zona(p, q):
    lab, hsv = p['lab'], p['hsv']
    m = np.zeros(lab.shape[:2], bool)
    m[q['y']:q['y'] + q['h'], q['x']:q['x'] + q['w']] = True
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    S = hsv[:, :, 1] / 255.0

    def cal(mm):
        return float(b[mm].mean() + 0.5 * a[mm].mean())   # b: azul->amarillo, a: verde->rojo

    r = {
        'brillo_zona': float(L[m].mean()), 'brillo_resto': float(L[~m].mean()),
        'calidez_zona': cal(m), 'calidez_resto': cal(~m),
        'saturacion_zona': float(S[m].mean()), 'saturacion_resto': float(S[~m].mean()),
        'contraste_zona': float(L[m].std()), 'contraste_total': float(L.std()),
    }
    r['dif_brillo'] = r['brillo_zona'] - r['brillo_resto']
    r['dif_calidez'] = r['calidez_zona'] - r['calidez_resto']
    r['dif_saturacion'] = r['saturacion_zona'] - r['saturacion_resto']
    rgb_medio = p['rgb'][m].reshape(-1, 3).mean(axis=0).astype(int)
    r['color_medio'] = '#%02x%02x%02x' % tuple(rgb_medio)
    r['color_nombre'] = nombre_color(rgb_medio)
    r['posicion'] = frase_posicion(q['celda']) if 'celda' in q else ''

    adjetivos, tiene = [], []
    if r['dif_brillo'] > 8:
        adjetivos.append('más luminosa')
    elif r['dif_brillo'] < -8:
        adjetivos.append('más oscura')
    if r['dif_calidez'] > 5:
        adjetivos.append('más cálida')
    elif r['dif_calidez'] < -5:
        adjetivos.append('más fría')
    if r['dif_saturacion'] > 0.08:
        tiene.append('colores más intensos')
    if r['contraste_zona'] > 1.3 * r['contraste_total']:
        tiene.append('mucho contraste interno')
    rasgos = adjetivos + ['con ' + t for t in tiene]
    r['rasgos'] = rasgos
    donde = (' (%s)' % r['posicion']) if r['posicion'] else ''
    inicio = 'La zona %d%s es de tono %s' % (q['id'], donde, r['color_nombre'])
    if rasgos:
        partes = []
        if adjetivos:
            partes.append('es ' + (adjetivos[0] if len(adjetivos) == 1 else ' y '.join(adjetivos)))
        if tiene:
            partes.append('tiene ' + ' y '.join(tiene))
        cuerpo = inicio + '. Frente al resto de la obra ' + ' y '.join(partes)
        cierre = []
        if r['dif_brillo'] > 8 or r['contraste_zona'] > 1.3 * r['contraste_total']:
            cierre.append('Una zona que contrasta con su entorno suele ser donde la mirada se detiene primero.')
        elif r['dif_brillo'] < -8:
            cierre.append('Una zona oscura puede leerse como sombra, profundidad o tensión, según la obra.')
        if abs(r['dif_calidez']) > 5:
            cierre.append('El choque cálido-frío con el resto puede orientar cómo se recorre la escena.')
        cierre.append('Es una descripción visual; cómo se vive esa zona depende de cada observador.')
        r['texto'] = cuerpo + '. ' + ' '.join(cierre)
    else:
        r['texto'] = (inicio + ', pero no se distingue mucho del resto en brillo, color o contraste; '
                      'su peso viene sobre todo de la posición.')
    return r


# ---------- referencia: la coleccion de obras ----------

def perfil_global(p):
    lab = p['lab']
    luz = luz_sombra(p)
    return {'brillo': float(lab[:, :, 0].mean()), 'contraste': float(lab[:, :, 0].std()),
            'calidez': float(lab[:, :, 2].mean() + 0.5 * lab[:, :, 1].mean()),
            'sombra': luz.get('sombra', float('nan'))}


def percentil(valor, serie):
    serie = np.asarray(serie, float)
    serie = serie[~np.isnan(serie)]
    return float((serie < valor).mean() * 100)


def comparar_con_coleccion(res, tabla):
    # res: salida de analizar_obra; tabla: DataFrame con perfil_global de cada obra de la coleccion
    mi = perfil_global(res['p'])
    return {k: {'valor': mi[k], 'percentil': percentil(mi[k], tabla[k])} for k in ('brillo', 'contraste', 'calidez', 'sombra')}


# ---------- 7, 8 y 9: foto, rostro y expresion ----------

class _CascadaSk:
    # mismo metodo que cv2.CascadeClassifier.detectMultiScale, usando el detector que trae scikit-image
    def __init__(self):
        from skimage.feature import Cascade
        from skimage import data as datos_sk
        self.det = Cascade(datos_sk.lbp_frontal_face_cascade_filename())

    def detectMultiScale(self, gris, escala=1.1, vecinos=5, minSize=(60, 60)):
        alto, ancho = gris.shape
        f = min(1.0, 640 / max(alto, ancho))
        chica = cv2.resize(gris, (int(ancho * f), int(alto * f))) if f < 1 else gris
        lado = max(int(minSize[0] * f), 20)
        # con pocos vecinos este detector inventa rostros; se exige un acuerdo alto y solo si no sale nada se baja
        for vec in (6, 4):
            r = self.det.detect_multi_scale(img=chica, scale_factor=1.2, step_ratio=1, min_size=(lado, lado),
                                            max_size=chica.shape, min_neighbor_number=vec)
            if len(r):
                break
        return np.array([[d['c'] / f, d['r'] / f, d['width'] / f, d['height'] / f] for d in r]).astype(int).reshape(-1, 4)


class _RedOrt:
    # por si cv2.dnn no abre el modelo: mismos metodos setInput y forward pero con onnxruntime
    def __init__(self, ruta):
        import onnxruntime as ort
        self.sesion = ort.InferenceSession(ruta)
        self.nombre = self.sesion.get_inputs()[0].name

    def setInput(self, x):
        self.x = x

    def forward(self):
        return self.sesion.run(None, {self.nombre: self.x})[0]


def decidir_expresion(logits, temp=1.0, umbral=0.55):
    # de los 8 logits de FER+ a: probabilidades, etiqueta (felicidad/tristeza/enojo o no concluyente) y confianza
    s = np.asarray(logits, float).ravel() / temp
    pr = np.exp(s - s.max())
    pr /= pr.sum()
    todas = {e: float(v) for e, v in zip(EXPRESIONES_FERPLUS, pr)}
    tres = {k: todas[k] for k in ('felicidad', 'tristeza', 'enojo')}
    mejor = max(tres, key=tres.get)
    if tres[mejor] < umbral or todas['neutral'] > tres[mejor]:
        return {'estado': 'expresion_incierta', 'etiqueta': 'no concluyente', 'confianza': float(tres[mejor]), 'probabilidades': todas}
    return {'estado': 'ok', 'etiqueta': mejor, 'confianza': float(tres[mejor]), 'probabilidades': todas}


class LectorRostro:
    def __init__(self, ruta_modelo, ruta_cascada=None, umbral=0.55, temperatura=1.0):
        self.cascada = None
        try:
            if ruta_cascada is None:
                ruta_cascada = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            c = cv2.CascadeClassifier(ruta_cascada)
            if not c.empty():
                self.cascada = c
        except Exception:
            pass
        if self.cascada is None:
            self.cascada = _CascadaSk()
        try:
            self.red = cv2.dnn.readNetFromONNX(ruta_modelo)
        except Exception:
            self.red = _RedOrt(ruta_modelo)
        self.umbral = umbral
        self.temp = temperatura

    def leer(self, foto):
        # la foto se procesa en memoria y no se guarda
        img, estado = abrir_imagen(foto, lado=1280)
        if img is None:
            return {'estado': 'foto_invalida'}
        gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        caras = self.cascada.detectMultiScale(gris, 1.1, 5, minSize=(60, 60))
        if len(caras) == 0:
            return {'estado': 'sin_rostro', 'n_rostros': 0}
        x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
        recorte = cv2.resize(gris[y:y + h, x:x + w], (64, 64)).astype(np.float32)
        self.red.setInput(recorte.reshape(1, 1, 64, 64))
        logits = np.asarray(self.red.forward()).ravel()
        res = decidir_expresion(logits, self.temp, self.umbral)
        res.update({'n_rostros': int(len(caras)), 'caja': [int(x), int(y), int(w), int(h)],
                    'recorte': recorte.astype(np.uint8), 'logits': logits.tolist()})
        if len(caras) > 1:
            res['aviso'] = 'hay %d rostros en la foto; se usó el más grande' % len(caras)
        return res


# ---------- 10: cruce ----------

def _sin_tildes(t):
    return ''.join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')

PALABRAS = {
    'felicidad': ['feliz', 'alegr', 'content', 'me gusta', 'agrad', 'bonit', 'lind'],
    'tristeza': ['trist', 'melanc', 'nostalg', 'soledad', 'pena', 'llor', 'desanim'],
    'enojo': ['enoj', 'rabia', 'molest', 'ira', 'fastid', 'indign'],
}

def emocion_declarada(texto):
    t = _sin_tildes(texto or '')
    cuenta = {e: sum(t.count(p) for p in ps) for e, ps in PALABRAS.items()}
    mejor = max(cuenta, key=cuenta.get)
    return mejor if cuenta[mejor] > 0 else None


def cruzar(zona, expresion, texto):
    declarada = emocion_declarada(texto)
    est = None if expresion is None else expresion.get('estado')
    out = {'zona': zona['texto'], 'texto_estudiante': texto, 'declarada': declarada}
    if expresion is None:
        out['estimada'] = None
        out['mensaje'] = 'No se cargó foto: se usa solo lo que escribió el estudiante.'
    elif est in ('ok', 'expresion_incierta'):
        out['estimada'] = expresion['etiqueta']
        out['confianza'] = expresion['confianza']
        if est == 'expresion_incierta':
            out['mensaje'] = 'La expresión no es concluyente; no se compara.'
        elif declarada is None:
            out['mensaje'] = 'Expresión estimada: %s. El texto no menciona una emoción reconocible.' % expresion['etiqueta']
        elif declarada == expresion['etiqueta']:
            out['mensaje'] = 'La expresión estimada coincide con lo declarado (%s).' % declarada
        else:
            out['mensaje'] = 'Declaró %s y la expresión estimada es %s. Son dos fuentes distintas y pueden no coincidir.' % (declarada, expresion['etiqueta'])
    else:
        out['estimada'] = None
        out['mensaje'] = {'sin_rostro': 'No se detectó un rostro en la foto; puede subir otra o continuar solo con texto.',
                          'foto_invalida': 'La foto no se pudo abrir; revise el formato (JPG o PNG).'}.get(est, 'No se pudo leer la foto.')
    return out


# ---------- integracion ----------

def analizar_obra(origen, k=5, n_puntos=4):
    img, estado = abrir_imagen(origen)
    if img is None:
        return {'estado': estado}
    p = preparar(img)
    mapa, cerca = saliencia(p)
    puntos = puntos_interes(mapa, cerca, n=n_puntos)
    res = {'estado': 'ok' if puntos else 'sin_puntos', 'alto': img.shape[0], 'ancho': img.shape[1],
           'p': p, 'paleta': paleta(p, k=k), 'luz': luz_sombra(p), 'saliencia': mapa, 'puntos': puntos}
    return res


def a_json(d):
    # saca los arreglos grandes para guardar o mostrar el resultado
    if isinstance(d, dict):
        return {k: a_json(v) for k, v in d.items() if k not in ('p', 'vista', 'mapa', 'saliencia', 'recorte', 'histograma')}
    if isinstance(d, (list, tuple)):
        return [a_json(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist() if d.size < 50 else None
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d