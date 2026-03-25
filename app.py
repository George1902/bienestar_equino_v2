import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── CONFIGURACIÓN ─────────────────────────────────────────
st.set_page_config(
    page_title="Bienestar Equino — Analizador de Riesgo Clínico Equino",
    page_icon="🐴",
    layout="centered"
)

# ── CARGA DE MODELOS ──────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    try:
        ruta_modelo = os.path.join("models", "modelo_equino.pkl")
        ruta_scaler = os.path.join("models", "scaler_equino.pkl")
        ruta_le     = os.path.join("models", "label_encoder.pkl")
        ruta_feat   = os.path.join("models", "features.pkl")

        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        with open(ruta_scaler, 'rb') as f:
            scaler = pickle.load(f)
        with open(ruta_le, 'rb') as f:
            le = pickle.load(f)
        with open(ruta_feat, 'rb') as f:
            features = pickle.load(f)

        return modelo, scaler, le, features
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, None

modelo, scaler, le, features = cargar_modelos()

if modelo is None:
    st.stop()

# ── HISTORIAL ─────────────────────────────────────────────
if 'historial' not in st.session_state:
    st.session_state.historial = []

# ── ENCABEZADO ────────────────────────────────────────────
st.title("🐴 Analizador de Riesgo Clínico Equino")
st.subheader("Herramienta de apoyo clínico veterinario")
st.info("Complete los datos clínicos y presione 'Analizar caballo'")
st.warning("""
⚠️ Esta herramienta es un apoyo basado en Machine Learning.
No reemplaza el diagnóstico veterinario profesional.
""")
st.divider()

# ── FORMULARIO ────────────────────────────────────────────
st.header("📋 Datos clínicos del caballo")

nombre_caballo = st.text_input(
    "🐎 Nombre del caballo",
    placeholder="Ej: Tornado, Relámpago, Luna...",
    help="Ingresa el nombre del caballo para identificar el análisis"
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Edad", [0, 1],
        format_func=lambda x: "Adulto" if x == 0 else "Joven")

    pulse = st.slider("Pulso (ppm)", 30, 184, 60,
        help="Rango normal: 28-44 ppm")

    rectal_temp = st.slider("Temperatura (°C)",
        35.0, 41.0, 38.2, 0.1,
        help="Rango normal: 37.5-38.5°C")

    respiratory_rate = st.slider("Frecuencia respiratoria",
        8, 96, 20,
        help="Rango normal: 8-16 rpm")

    surgery = st.selectbox("¿Cirugía?", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí")

with col2:
    pain = st.selectbox("Nivel de dolor", [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Sin dolor",
            1: "Deprimido",
            2: "Dolor leve",
            3: "Dolor severo",
            4: "Dolor extremo"
        }[x])

    peristalsis = st.selectbox("Peristalsis", [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Ausente",
            1: "Hipomotil",
            2: "Normal",
            3: "Hipermotil"
        }[x])

    packed_cell_volume = st.slider("Volumen celular (%)",
        23, 75, 45)

    total_protein = st.slider("Proteína total (g/dl)",
        3.0, 89.0, 7.5, 0.1)

    abdominal_distention = st.selectbox(
        "Distensión abdominal", [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Ninguna",
            1: "Leve",
            2: "Moderada",
            3: "Severa"
        }[x])

st.divider()

# ── FUNCIONES ─────────────────────────────────────────────
def calcular_bienestar(pulse, rectal_temp, pain, peristalsis):
    score = 0
    if pulse <= 44:   score += 3
    elif pulse <= 60: score += 2
    elif pulse <= 80: score += 1
    if 37.5 <= rectal_temp <= 38.5:   score += 2
    elif 37.0 <= rectal_temp <= 39.0: score += 1
    if pain == 0:   score += 3
    elif pain == 1: score += 2
    elif pain == 2: score += 1
    if peristalsis == 2: score += 2
    elif peristalsis == 1: score += 1
    return score

def nivel_bienestar(score):
    if score >= 8:   return "Alto",     "🟢"
    elif score >= 5: return "Moderado", "🟡"
    elif score >= 2: return "Bajo",     "🟠"
    else:            return "Crítico",  "🔴"

# ── BOTÓN DE ANÁLISIS ─────────────────────────────────────
if st.button("🔍 Analizar caballo", use_container_width=True,
             type="primary"):

    nombre = nombre_caballo.strip() \
             if nombre_caballo.strip() else "Sin nombre"

    entrada = {f: 0 for f in features}
    entrada.update({
        'age'                 : age,
        'pulse'               : pulse,
        'rectal_temp'         : rectal_temp,
        'respiratory_rate'    : respiratory_rate,
        'surgery'             : surgery,
        'pain'                : pain,
        'peristalsis'         : peristalsis,
        'packed_cell_volume'  : packed_cell_volume,
        'total_protein'       : total_protein,
        'abdominal_distention': abdominal_distention
    })

    idx = calcular_bienestar(pulse, rectal_temp, pain, peristalsis)
    entrada['indice_bienestar'] = idx

    try:
        X         = pd.DataFrame([entrada])[features]
        X_scaled  = scaler.transform(X)
        pred      = modelo.predict(X_scaled)
        proba     = modelo.predict_proba(X_scaled)[0]
        resultado = le.inverse_transform(pred.astype(int))[0]
        nivel, emoji = nivel_bienestar(idx)

        # Guardar en historial
        st.session_state.historial.append({
            'Nombre'      : nombre,
            'Pronostico'  : '✅ Sobrevive'   if resultado == 'lived'
                            else '❌ Alto riesgo' if resultado == 'died'
                            else '⚠️ Eutanasia',
            'Probabilidad': f"{max(proba)*100:.1f}%",
            'Bienestar'   : f"{idx}/9",
            'Nivel'       : f"{emoji} {nivel}",
            'Pulso'       : f"{pulse} ppm",
            'Temperatura' : f"{rectal_temp}°C",
            'Dolor'       : {
                0: "Sin dolor",
                1: "Deprimido",
                2: "Dolor leve",
                3: "Dolor severo",
                4: "Dolor extremo"
            }[pain]
        })

        st.divider()
        st.header(f"📊 Resultado — {nombre}")

        # Resultado principal
        if resultado == 'lived':
            st.success("## ✅ PRONÓSTICO: SOBREVIVIRÁ")
        elif resultado == 'died':
            st.error("## ❌ PRONÓSTICO: ALTO RIESGO DE MUERTE")
        else:
            st.warning("## ⚠️ PRONÓSTICO: CONSIDERAR EUTANASIA")

        # Probabilidades
        st.subheader("📈 Probabilidades")
        col1, col2, col3 = st.columns(3)
        nombres_clases = {
            'lived'     : '✅ Sobrevive',
            'died'      : '❌ Muere',
            'euthanized': '⚠️ Eutanasia'
        }
        for col, clase, p in zip(
                [col1, col2, col3], le.classes_, proba):
            col.metric(nombres_clases[clase], f"{p*100:.1f}%")

        # Índice de bienestar
        st.subheader("🏥 Índice de Bienestar")
        col1, col2 = st.columns(2)
        col1.metric("Puntuación", f"{idx}/9")
        col2.metric("Nivel", f"{emoji} {nivel}")
        st.progress(idx / 9)

        # Alertas clínicas
        st.subheader("⚠️ Alertas clínicas")
        alertas = []

        if pulse > 80:
            alertas.append(
                "🔴 Pulso muy elevado — estrés cardiovascular severo")
        elif pulse > 60:
            alertas.append(
                "🟡 Pulso elevado — monitorear de cerca")

        if rectal_temp > 39.0:
            alertas.append(
                "🔴 Temperatura alta — posible infección")
        elif rectal_temp < 37.0:
            alertas.append(
                "🔴 Temperatura baja — hipotermia")

        if pain >= 3:
            alertas.append(
                "🔴 Dolor severo — requiere atención inmediata")

        if peristalsis == 0:
            alertas.append(
                "🔴 Peristalsis ausente — riesgo de cólico")

        if total_protein > 8.5:
            alertas.append(
                "🟡 Proteína elevada — posible deshidratación")

        if alertas:
            for alerta in alertas:
                st.warning(alerta)
        else:
            st.success("✅ Sin alertas clínicas críticas detectadas")

    except Exception as e:
        st.error(f"Error en predicción: {e}")

    st.divider()
    st.caption("""
    ⚠️ Este modelo es una herramienta de apoyo clínico con
    65% de accuracy. No reemplaza el diagnóstico veterinario
    profesional. Desarrollado por Jorge Ojeda — ONE Alura LATAM 2026.
    """)

# ── HISTORIAL DE CABALLOS ANALIZADOS ─────────────────────
if st.session_state.historial:
    st.divider()
    st.header("📋 Historial de caballos analizados")
    st.caption(f"Total analizados en esta sesión: "
               f"{len(st.session_state.historial)}")

    df_historial = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_historial, use_container_width=True)

    col1, col2 = st.columns(2)

    # Resumen rápido
    with col1:
        sobreviven = sum(
            1 for h in st.session_state.historial
            if '✅' in h['Pronostico']
        )
        st.metric("✅ Sobreviven", sobreviven)

    with col2:
        en_riesgo = sum(
            1 for h in st.session_state.historial
            if '❌' in h['Pronostico'] or '⚠️' in h['Pronostico']
        )
        st.metric("⚠️ En riesgo", en_riesgo)

    # Botón limpiar
    if st.button("🗑️ Limpiar historial",
                 use_container_width=True):
        st.session_state.historial = []
        st.rerun()

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── CONFIGURACIÓN ─────────────────────────────────────────
st.set_page_config(
    page_title="Analizador de Riesgo Clínico Equino",
    page_icon="🐴",
    layout="centered"
)

# ── CARGA DE MODELOS ──────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    try:
        ruta_modelo = os.path.join("models", "modelo_equino.pkl")
        ruta_scaler = os.path.join("models", "scaler_equino.pkl")
        ruta_le     = os.path.join("models", "label_encoder.pkl")
        ruta_feat   = os.path.join("models", "features.pkl")

        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        with open(ruta_scaler, 'rb') as f:
            scaler = pickle.load(f)
        with open(ruta_le, 'rb') as f:
            le = pickle.load(f)
        with open(ruta_feat, 'rb') as f:
            features = pickle.load(f)

        return modelo, scaler, le, features
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, None

modelo, scaler, le, features = cargar_modelos()

if modelo is None:
    st.stop()

# ── ENCABEZADO ────────────────────────────────────────────
st.title("🐴 Analizador de Riesgo Clínico Equino")
st.subheader("Herramienta de apoyo clínico veterinario")
st.info("Complete los datos clínicos y presione 'Analizar caballo'")
st.warning("""
⚠️ Esta herramienta es un apoyo basado en Machine Learning.
No reemplaza el diagnóstico veterinario profesional.
""")
st.divider()

# ── FORMULARIO ────────────────────────────────────────────
st.header("📋 Datos clínicos del caballo")

# Nombre del caballo
nombre_caballo = st.text_input(
    "🐎 Nombre del caballo",
    placeholder="Ej: Tornado, Relámpago, Luna...",
    help="Ingresa el nombre del caballo para identificar el análisis"
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Edad", [0, 1],
        format_func=lambda x: "Adulto" if x == 0 else "Joven")

    pulse = st.slider("Pulso (ppm)", 30, 184, 60,
        help="Rango normal: 28-44 ppm")

    rectal_temp = st.slider("Temperatura (°C)",
        35.0, 41.0, 38.2, 0.1,
        help="Rango normal: 37.5-38.5°C")

    respiratory_rate = st.slider("Frecuencia respiratoria",
        8, 96, 20,
        help="Rango normal: 8-16 rpm")

    surgery = st.selectbox("¿Cirugía?", [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí")

with col2:
    pain = st.selectbox("Nivel de dolor", [0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "Sin dolor",
            1: "Deprimido",
            2: "Dolor leve",
            3: "Dolor severo",
            4: "Dolor extremo"
        }[x])

    peristalsis = st.selectbox("Peristalsis", [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Ausente",
            1: "Hipomotil",
            2: "Normal",
            3: "Hipermotil"
        }[x])

    packed_cell_volume = st.slider("Volumen celular (%)",
        23, 75, 45)

    total_protein = st.slider("Proteína total (g/dl)",
        3.0, 89.0, 7.5, 0.1)

    abdominal_distention = st.selectbox(
        "Distensión abdominal", [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Ninguna",
            1: "Leve",
            2: "Moderada",
            3: "Severa"
        }[x])

st.divider()

# ── FUNCIONES ─────────────────────────────────────────────
def calcular_bienestar(pulse, rectal_temp, pain, peristalsis):
    score = 0
    if pulse <= 44:   score += 3
    elif pulse <= 60: score += 2
    elif pulse <= 80: score += 1
    if 37.5 <= rectal_temp <= 38.5:   score += 2
    elif 37.0 <= rectal_temp <= 39.0: score += 1
    if pain == 0:   score += 3
    elif pain == 1: score += 2
    elif pain == 2: score += 1
    if peristalsis == 2: score += 2
    elif peristalsis == 1: score += 1
    return score

def nivel_bienestar(score):
    if score >= 8: return "Alto",     "🟢"
    elif score >= 5: return "Moderado", "🟡"
    elif score >= 2: return "Bajo",     "🟠"
    else:            return "Crítico",  "🔴"

# ── BOTÓN DE ANÁLISIS ─────────────────────────────────────
if st.button("🔍 Analizar caballo", use_container_width=True,
             type="primary"):

    # Validar nombre
    nombre = nombre_caballo.strip() if nombre_caballo.strip() else "Sin nombre"

    entrada = {f: 0 for f in features}
    entrada.update({
        'age'                 : age,
        'pulse'               : pulse,
        'rectal_temp'         : rectal_temp,
        'respiratory_rate'    : respiratory_rate,
        'surgery'             : surgery,
        'pain'                : pain,
        'peristalsis'         : peristalsis,
        'packed_cell_volume'  : packed_cell_volume,
        'total_protein'       : total_protein,
        'abdominal_distention': abdominal_distention
    })

    idx = calcular_bienestar(pulse, rectal_temp, pain, peristalsis)
    entrada['indice_bienestar'] = idx

    try:
        X        = pd.DataFrame([entrada])[features]
        X_scaled = scaler.transform(X)
        pred     = modelo.predict(X_scaled)
        proba    = modelo.predict_proba(X_scaled)[0]
        resultado = le.inverse_transform(pred.astype(int))[0]
        nivel, emoji = nivel_bienestar(idx)

        st.divider()

        # Nombre del caballo en el resultado
        st.header(f"📊 Resultado — {nombre}")

        # Resultado principal
        if resultado == 'lived':
            st.success("## ✅ PRONÓSTICO: SOBREVIVIRÁ")
        elif resultado == 'died':
            st.error("## ❌ PRONÓSTICO: ALTO RIESGO DE MUERTE")
        else:
            st.warning("## ⚠️ PRONÓSTICO: CONSIDERAR EUTANASIA")

        # Probabilidades
        st.subheader("📈 Probabilidades")
        col1, col2, col3 = st.columns(3)
        nombres_clases = {
            'lived'     : '✅ Sobrevive',
            'died'      : '❌ Muere',
            'euthanized': '⚠️ Eutanasia'
        }
        for col, clase, p in zip(
            [col1, col2, col3], le.classes_, proba):
            col.metric(nombres_clases[clase], f"{p*100:.1f}%")

        # Índice de bienestar
        st.subheader("🏥 Índice de Bienestar")
        col1, col2 = st.columns(2)
        col1.metric("Puntuación", f"{idx}/9")
        col2.metric("Nivel", f"{emoji} {nivel}")
        st.progress(idx / 9)

        # Alertas clínicas — solo aparecen al presionar el botón
        st.subheader("⚠️ Alertas clínicas")
        alertas = []

        if pulse > 80:
            alertas.append(
                "🔴 Pulso muy elevado — estrés cardiovascular severo")
        elif pulse > 60:
            alertas.append(
                "🟡 Pulso elevado — monitorear de cerca")

        if rectal_temp > 39.0:
            alertas.append(
                "🔴 Temperatura alta — posible infección")
        elif rectal_temp < 37.0:
            alertas.append(
                "🔴 Temperatura baja — hipotermia")

        if pain >= 3:
            alertas.append(
                "🔴 Dolor severo — requiere atención inmediata")

        if peristalsis == 0:
            alertas.append(
                "🔴 Peristalsis ausente — riesgo de cólico")

        if total_protein > 8.5:
            alertas.append(
                "🟡 Proteína elevada — posible deshidratación")

        if alertas:
            for alerta in alertas:
                st.warning(alerta)
        else:
            st.success("✅ Sin alertas clínicas críticas detectadas")

    except Exception as e:
        st.error(f"Error en predicción: {e}")

    st.divider()
    st.caption("""
    ⚠️ Este modelo es una herramienta de apoyo clínico con
    65% de accuracy. No reemplaza el diagnóstico veterinario
    profesional. Desarrollado por Jorge Ojeda.
    """)
