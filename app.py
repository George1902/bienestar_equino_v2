import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# CONFIGURACION
st.set_page_config(
    page_title="Analizador de Riesgo Clinico Equino",
    page_icon="🐴",
    layout="centered"
)

# ESTILO CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: clamp(4rem, 10vw, 12rem);
    font-weight: 900;
    margin-top: -20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main-subtitle {
    text-align: center;
    font-size: clamp(2rem, 5vw, 6rem);
    color: #6c757d;
    margin-bottom: 1rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# CARGA DE MODELOS
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

# HISTORIAL
if 'historial' not in st.session_state:
    st.session_state.historial = []

# ENCABEZADO
st.markdown(
    '<p class="main-title">\U0001f434 Analizador de Riesgo Clinico Equino</p>',
    unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Herramienta de apoyo clinico veterinario</p>',
    unsafe_allow_html=True)
st.warning(
    "\u26a0\ufe0f Esta herramienta es un apoyo basado en Machine Learning. "
    "No reemplaza el diagnostico veterinario profesional.")
st.divider()

# TABS
tab1, tab2 = st.tabs(["\U0001f50d Analizar caballo",
                      "\U0001f4cb Historial"])

# TAB 1 - FORMULARIO
with tab1:

    st.header("\U0001f4cb Datos clinicos del caballo")

    nombre_caballo = st.text_input(
        "\U0001f40e Nombre del caballo",
        placeholder="Ej: Tornado, Relampago, Luna...",
        help="Ingresa el nombre del caballo"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Edad", [0, 1],
            format_func=lambda x: "Adulto" if x == 0 else "Joven")

        pulse = st.slider("Pulso (ppm)", 30, 184, 60,
            help="Rango normal: 28-44 ppm")

        rectal_temp = st.slider("Temperatura (C)",
            35.0, 41.0, 38.2, 0.1,
            help="Rango normal: 37.5-38.5 C")

        respiratory_rate = st.slider("Frecuencia respiratoria",
            8, 96, 20,
            help="Rango normal: 8-16 rpm")

        surgery = st.selectbox("Cirugia?", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Si")

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

        total_protein = st.slider("Proteina total (g/dl)",
            3.0, 89.0, 7.5, 0.1)

        abdominal_distention = st.selectbox(
            "Distension abdominal", [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Ninguna",
                1: "Leve",
                2: "Moderada",
                3: "Severa"
            }[x])

    st.divider()

    # FUNCIONES
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
        if score >= 8:   return "Alto",     "\U0001f7e2"
        elif score >= 5: return "Moderado", "\U0001f7e1"
        elif score >= 2: return "Bajo",     "\U0001f7e0"
        else:            return "Critico",  "\U0001f534"

    # BOTON
    if st.button("\U0001f50d Analizar caballo",
                 use_container_width=True,
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

        idx = calcular_bienestar(
            pulse, rectal_temp, pain, peristalsis)
        entrada['indice_bienestar'] = idx

        try:
            X         = pd.DataFrame([entrada])[features]
            X_scaled  = scaler.transform(X)
            pred      = modelo.predict(X_scaled)
            proba     = modelo.predict_proba(X_scaled)[0]
            resultado = le.inverse_transform(
                pred.astype(int))[0]
            nivel, icono = nivel_bienestar(idx)

            # Guardar en historial
            st.session_state.historial.append({
                'Nombre'      : nombre,
                'Pronostico'  : '\u2705 Sobrevive'
                                if resultado == 'lived'
                                else '\u274c Alto riesgo'
                                if resultado == 'died'
                                else '\u26a0\ufe0f Eutanasia',
                'Probabilidad': f"{max(proba)*100:.1f}%",
                'Bienestar'   : f"{idx}/9",
                'Nivel'       : f"{icono} {nivel}",
                'Pulso'       : f"{pulse} ppm",
                'Temperatura' : f"{rectal_temp} C",
                'Dolor'       : {
                    0: "Sin dolor",
                    1: "Deprimido",
                    2: "Dolor leve",
                    3: "Dolor severo",
                    4: "Dolor extremo"
                }[pain]
            })

            st.divider()
            st.header(f"\U0001f4ca Resultado - {nombre}")

            # Resultado principal
            if resultado == 'lived':
                st.success(
                    "\u2705 PRONOSTICO: SOBREVIVIRA")
            elif resultado == 'died':
                st.error(
                    "\u274c PRONOSTICO: ALTO RIESGO DE MUERTE")
            else:
                st.warning(
                    "\u26a0\ufe0f PRONOSTICO: CONSIDERAR EUTANASIA")

            # Probabilidades
            st.subheader("\U0001f4c8 Probabilidades")
            col1, col2, col3 = st.columns(3)
            nombres_clases = {
                'lived'     : '\u2705 Sobrevive',
                'died'      : '\u274c Muere',
                'euthanized': '\u26a0\ufe0f Eutanasia'
            }
            for col, clase, p in zip(
                    [col1, col2, col3], le.classes_, proba):
                col.metric(nombres_clases[clase],
                           f"{p*100:.1f}%")

            # Indice de bienestar
            st.subheader("\U0001f3e5 Indice de Bienestar")
            col1, col2 = st.columns(2)
            col1.metric("Puntuacion", f"{idx}/9")
            col2.metric("Nivel", f"{icono} {nivel}")
            st.progress(idx / 9)

            # Alertas clinicas
            st.subheader("\u26a0\ufe0f Alertas clinicas")
            alertas = []

            if pulse > 80:
                alertas.append(
                    "\U0001f534 Pulso muy elevado - "
                    "estres cardiovascular severo")
            elif pulse > 60:
                alertas.append(
                    "\U0001f7e1 Pulso elevado - "
                    "monitorear de cerca")

            if rectal_temp > 39.0:
                alertas.append(
                    "\U0001f534 Temperatura alta - "
                    "posible infeccion")
            elif rectal_temp < 37.0:
                alertas.append(
                    "\U0001f534 Temperatura baja - hipotermia")

            if pain >= 3:
                alertas.append(
                    "\U0001f534 Dolor severo - "
                    "requiere atencion inmediata")

            if peristalsis == 0:
                alertas.append(
                    "\U0001f534 Peristalsis ausente - "
                    "riesgo de colico")

            if total_protein > 8.5:
                alertas.append(
                    "\U0001f7e1 Proteina elevada - "
                    "posible deshidratacion")

            if alertas:
                for alerta in alertas:
                    st.warning(alerta)
            else:
                st.success(
                    "\u2705 Sin alertas clinicas criticas")

            st.info(
                "\U0001f4a1 Revisa la pestana Historial "
                "para ver todos los caballos analizados")

        except Exception as e:
            st.error(f"Error en prediccion: {e}")

        st.divider()
        st.caption(
            "\u26a0\ufe0f Este modelo es una herramienta de "
            "apoyo clinico con 65% de accuracy. No reemplaza "
            "el diagnostico veterinario profesional. "
            "Desarrollado por Jorge Ojeda.")

# TAB 2 - HISTORIAL
with tab2:

    st.header("\U0001f4cb Historial de caballos analizados")

    if not st.session_state.historial:
        st.info(
            "Aun no has analizado ningun caballo. "
            "Ve a la pestana Analizar caballo.")
    else:
        st.caption(
            f"Total analizados en esta sesion: "
            f"{len(st.session_state.historial)}")

        col1, col2, col3 = st.columns(3)
        sobreviven = sum(
            1 for h in st.session_state.historial
            if 'Sobrevive' in h['Pronostico'])
        en_riesgo = sum(
            1 for h in st.session_state.historial
            if 'Alto riesgo' in h['Pronostico'])
        eutanasia = sum(
            1 for h in st.session_state.historial
            if 'Eutanasia' in h['Pronostico'])

        col1.metric("\u2705 Sobreviven", sobreviven)
        col2.metric("\u274c Alto riesgo", en_riesgo)
        col3.metric("\u26a0\ufe0f Eutanasia", eutanasia)

        st.divider()

        df_historial = pd.DataFrame(
            st.session_state.historial)
        st.dataframe(df_historial,
                     use_container_width=True)

        st.divider()

        if st.button("\U0001f5d1\ufe0f Limpiar historial",
                     use_container_width=True):
            st.session_state.historial = []
            st.rerun()
