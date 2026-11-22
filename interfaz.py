# interfaz.py
# Interfaz minima en Gradio, una pestaña por fase: carga, exploracion, registro, sintesis.
import json, tempfile
import numpy as np
import gradio as gr
import lectura_obras as lo
import modelo_epoca as me

AVISOS = {
    'imagen_invalida': 'No se pudo abrir la imagen. Suba un JPG o PNG.',
    'imagen_muy_pequena': 'La imagen es muy pequeña (mínimo 64 px por lado).',
    'sin_puntos': 'No se encontraron zonas que destaquen en esta obra. Pruebe con otra imagen.',
}
LENTES = ['Original + zona', 'Paleta reducida', 'Luz y sombra']


NOTA_EXPRESION = ('**Nota:** la expresión facial estimada es una referencia visual de la foto; el sistema no mide emociones '
                  'y lo que usted escribió es la fuente principal.')


def procesar_registro(lector, estado, acepta, texto, foto):
    # paso 3: guarda el texto; la foto solo se lee si hay consentimiento y nunca se escribe en disco
    if not estado:
        return 'Primero cargue y analice una obra.'
    estado['texto'] = texto
    estado['expresion'] = None
    if foto is not None:
        if not acepta:
            return 'Para usar la foto debe aceptar el consentimiento. Se guardó solo su texto.'
        estado['expresion'] = lector.leer(foto)
    e = estado['expresion']
    if e is None:
        return 'Registrado solo el texto.'
    if e['estado'] in ('ok', 'expresion_incierta'):
        m = 'Foto procesada en memoria (no se guarda). Expresión estimada: **%s** (%.2f).' % (e['etiqueta'], e['confianza'])
        if e.get('aviso'):
            m += ' ' + e['aviso'] + '.'
        return m
    return {'sin_rostro': 'No se detectó un rostro. Pruebe con otra foto de frente y con buena luz.',
            'foto_invalida': 'No se pudo abrir la foto. Use JPG o PNG.'}[e['estado']]


def sintesis_texto(estado):
    # paso 4: texto de la sintesis y registro exportable
    r = estado['analisis']
    q = r['puntos'][estado['zona_id'] - 1]
    z = lo.leer_zona(r['p'], q)
    c = lo.cruzar(z, estado['expresion'], estado['texto'])
    md = '**Zona %d.** %s\n\n**Lo que escribió:** %s\n\n**Cruce:** %s\n\n%s' % (
        estado['zona_id'], z['texto'], estado['texto'] or '(sin texto)', c['mensaje'], NOTA_EXPRESION)
    return md, lo.a_json({'zona': q, 'lectura': z, 'cruce': c})


def crear_interfaz(lector, tabla_coleccion=None, extractor=None, modelo_epoca=None, umbral_epoca=0.5, predictor_epoca=None):
    # predictor_epoca: funcion rgb -> dict (como me.predecir_epoca). Si no se da, se usa extractor + modelo_epoca.

    def paso1(img, k):
        if img is None:
            return None, 'Cargue una imagen de una pintura.', gr.Dropdown(choices=[], value=None), None
        r = lo.analizar_obra(img, k=int(k))
        if r['estado'] in AVISOS:
            return None, AVISOS[r['estado']], gr.Dropdown(choices=[], value=None), None
        luz = r['luz']
        texto = ['**Obra cargada** (%d x %d px). Se encontraron %d zonas de interés.' % (r['ancho'], r['alto'], len(r['puntos']))]
        if luz['estado'] == 'ok':
            texto.append('Sombras %.0f %% · medios %.0f %% · luces %.0f %% · contraste %.1f' %
                         (100 * luz['sombra'], 100 * luz['medios'], 100 * luz['luces'], luz['contraste']))
        if tabla_coleccion is not None:
            c = lo.comparar_con_coleccion(r, tabla_coleccion)
            texto.append('Frente a las %d obras de la colección: más oscura que el %.0f %% y con más contraste que el %.0f %%.' %
                         (len(tabla_coleccion), 100 - c['brillo']['percentil'], c['contraste']['percentil']))
        pred = predictor_epoca
        if pred is None and extractor is not None and modelo_epoca is not None:
            pred = lambda rgb: me.predecir_epoca(modelo_epoca, extractor.uno(rgb), umbral_epoca)
        if pred is not None:
            e = pred(r['p']['rgb'])
            if e['estado'] == 'ok':
                texto.append('Época sugerida: %s (confianza %.2f). Es una pista, no una atribución.' % (e['epoca'], e['confianza']))
            else:
                texto.append('Época: no concluyente (confianza %.2f), no se afirma ninguna.' % e['confianza'])
        ids = [str(q['id']) for q in r['puntos']]
        estado = {'analisis': r, 'texto': None, 'expresion': None, 'zona_id': 1}
        return lo.marcar_puntos(r['p']['rgb'], r['puntos']), '\n\n'.join(texto), gr.Dropdown(choices=ids, value=ids[0]), estado

    def paso2(estado, zona, lente, alfa):
        if not estado:
            return None, 'Primero cargue y analice una obra.'
        r = estado['analisis']
        q = r['puntos'][int(zona) - 1]
        estado['zona_id'] = int(zona)
        base = r['p']['rgb']
        if lente == 'Paleta reducida':
            vista = r['paleta']['vista']
        elif lente == 'Luz y sombra':
            vista = r['luz']['vista'] if r['luz']['estado'] == 'ok' else base
        else:
            vista = base
        img = lo.mezclar(base, vista, alfa) if vista is not base else base
        img = lo.marcar_puntos(img, r['puntos'], elegido=q['id'])
        z = lo.leer_zona(r['p'], q)
        txt = z['texto'] + '\n\nBrillo %.0f vs %.0f · calidez %.1f vs %.1f · saturación %.2f vs %.2f (zona vs resto).' % (
            z['brillo_zona'], z['brillo_resto'], z['calidez_zona'], z['calidez_resto'], z['saturacion_zona'], z['saturacion_resto'])
        return img, txt

    def paso3(estado, acepta, texto, foto):
        return procesar_registro(lector, estado, acepta, texto, foto)

    def paso4(estado):
        if not estado:
            return 'Primero cargue y analice una obra.', None, None
        md, reg = sintesis_texto(estado)
        archivo = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(reg, archivo, ensure_ascii=False, indent=2, default=str)
        archivo.close()
        return md, reg, archivo.name

    with gr.Blocks() as demo:
        gr.Markdown('## Laboratorio de lectura de pinturas')
        estado = gr.State(None)
        with gr.Tab('1. Carga y detección'):
            entrada = gr.Image(label='Pintura', type='numpy')
            k = gr.Slider(3, 8, value=5, step=1, label='Colores de la paleta')
            b1 = gr.Button('Analizar obra')
            salida1 = gr.Image(label='Zonas de interés')
            info1 = gr.Markdown()
        with gr.Tab('2. Exploración de zonas'):
            zona = gr.Dropdown(label='Zona', choices=[])
            lente = gr.Radio(LENTES, value=LENTES[0], label='Lente')
            alfa = gr.Slider(0, 1, value=0.6, label='Opacidad de la vista transformada')
            b2 = gr.Button('Ver zona')
            salida2 = gr.Image(label='Vista')
            info2 = gr.Markdown()
        with gr.Tab('3. Registro'):
            gr.Markdown('Mire la zona elegida y cuente qué siente. La foto es opcional; se procesa en memoria y no se guarda.')
            texto = gr.Textbox(label='¿Qué siente al ver esta zona?', lines=3)
            acepta = gr.Checkbox(label='Acepto que se procese mi foto para estimar la expresión facial')
            foto = gr.Image(label='Foto (opcional)', type='numpy', sources=['upload', 'webcam'])
            b3 = gr.Button('Registrar')
            info3 = gr.Markdown()
        with gr.Tab('4. Síntesis'):
            b4 = gr.Button('Ver síntesis')
            info4 = gr.Markdown()
            datos = gr.JSON(label='Registro')
            archivo = gr.File(label='Descargar registro')
        b1.click(paso1, [entrada, k], [salida1, info1, zona, estado])
        b2.click(paso2, [estado, zona, lente, alfa], [salida2, info2])
        b3.click(paso3, [estado, acepta, texto, foto], [info3])
        b4.click(paso4, [estado], [info4, datos, archivo])
    return demo