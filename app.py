import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# CONFIGURACION
st.set_page_config(
    page_title="Analizador de Riesgo Clínico Equino",
    page_icon="🐴",
    layout="centered"
)

# ESTILO CSS (TAMAÑO EQUILIBRADO)
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: clamp(2.5rem, 5vw, 4rem) !important;
    font-weight: 800;
    margin-bottom: 0;
    line-height: 1.2;
}

.main-subtitle {
    text-align: center;
    font-size: clamp(1.2rem, 2vw, 1.8rem) !important;
    color: #6c757d;
    margin-top: 0.2rem;
    margin-bottom: 1.2rem;
    font-weight: 400;
    opacity: 0.85;
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
    return min(score, 9)  # Limitar máximo a 9

def nivel_bienestar(score):
    if score >= 8:   return "Alto",     "\U0001f7e2"
    elif score >= 5: return "Moderado", "\U0001f7e1"
    elif score >= 2: return "Bajo",     "\U0001f7e0"
    else:            return "Critico",  "\U0001f534"

def motor_clinico(pulse, rectal_temp, pain, peristalsis,
                  total_protein, respiratory_rate, nivel):

    sistema_gi   = 0
    sistema_hemo = 0
    sistema_sist = 0
    hallazgos    = []

    # SISTEMA GASTROINTESTINAL
    if peristalsis == 0:
        sistema_gi += 3
        hallazgos.append("ausencia de peristalsis")
    elif peristalsis == 1:
        sistema_gi += 2
        hallazgos.append("hipomotilidad intestinal")

    if pain >= 3:
        sistema_gi += 2
        hallazgos.append("dolor abdominal significativo")

    # SISTEMA HEMODINAMICO
    if pulse > 80:
        sistema_hemo += 3
        hallazgos.append("taquicardia severa")
    elif pulse > 60:
        sistema_hemo += 2
        hallazgos.append("taquicardia moderada")

    if total_protein > 8.5:
        sistema_hemo += 2
        hallazgos.append("hemoconcentracion")

    # SISTEMA SISTEMICO
    if rectal_temp > 39:
        sistema_sist += 2
        hallazgos.append("hipertermia")
    elif rectal_temp < 37:
        sistema_sist += 2
        hallazgos.append("hipotermia")

    if respiratory_rate > 20:
        sistema_sist += 1
        hallazgos.append("taquipnea")

    # SISTEMA DOMINANTE
    sistemas = {
        "gastrointestinal": sistema_gi,
        "hemodinamico"    : sistema_hemo,
        "sistemico"       : sistema_sist
    }

    sistema_dominante = max(sistemas, key=sistemas.get)

    # GRAVEDAD
    gravedad_total = sistema_gi + sistema_hemo + sistema_sist

    if gravedad_total >= 7:
        gravedad = "alta"
    elif gravedad_total >= 4:
        gravedad = "moderada"
    else:
        gravedad = "leve"

    # RESUMEN CLINICO
    resumen = f"Paciente con compromiso {gravedad} "
    resumen += f"con predominio del sistema {sistema_dominante}. "

    if hallazgos:
        resumen += "Se identifican hallazgos relevantes como: "
        resumen += ", ".join(hallazgos) + "."

    return resumen, sistema_dominante, gravedad

# ENCABEZADO
st.markdown(
    '<p class="main-title">\U0001f434 Analizador de Riesgo Clínico Equino</p>',
    unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Herramienta de apoyo clínico veterinario</p>',
    unsafe_allow_html=True)
st.warning(
    "\u26a0\ufe0f Esta herramienta es un apoyo basado en "
    "Machine Learning. No reemplaza el diagnóstico "
    "veterinario profesional.")
st.divider()

# TABS
tab1, tab2 = st.tabs(["\U0001f50d Analizar caballo",
                      "\U0001f4cb Historial"])

# ══════════════════════════════════════════════════════════
# TAB 1 - FORMULARIO
# ══════════════════════════════════════════════════════════
with tab1:

    st.header("\U0001f4cb Datos clínicos del caballo")

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

            # Motor clinico
            resumen, sistema, gravedad = motor_clinico(
                pulse, rectal_temp, pain, peristalsis,
                total_protein, respiratory_rate, nivel)

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
                'Sistema'     : sistema,
                'Gravedad'    : gravedad,
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
                st.success("\u2705 PRONÓSTICO: SOBREVIVIRÁ")
            elif resultado == 'died':
                st.error("\u274c PRONÓSTICO: ALTO RIESGO DE MUERTE")
            else:
                st.warning(
                    "\u26a0\ufe0f PRONÓSTICO: CONSIDERAR EUTANASIA")

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
            st.subheader("\U0001f3e5 Índice de Bienestar")
            col1, col2 = st.columns(2)
            col1.metric("Puntuacion", f"{idx}/9")
            col2.metric("Nivel", f"{icono} {nivel}")
            st.progress(idx / 9)

            # Resumen clinico avanzado
            st.subheader("\U0001f9fe Resumen clínico avanzado")
            st.info(resumen)

            # Sistema predominante
            st.subheader("\U0001f9e0 Sistema predominante")
            if sistema == "gastrointestinal":
                st.error(
                    "\U0001f534 Compromiso gastrointestinal predominante")
            elif sistema == "hemodinamico":
                st.warning(
                    "\U0001f7e1 Compromiso hemodinamico predominante")
            else:
                st.info(
                    "\U0001f535 Compromiso sistemico predominante")

            # Interpretacion clinica
            st.subheader("\U0001f4ca Interpretacion clínica")
            if gravedad == "alta":
                st.write(
                    "El perfil clínico indica un cuadro de alta "
                    "complejidad, con multiples sistemas comprometidos. "
                    "Se recomienda atencion veterinaria inmediata.")
            elif gravedad == "moderada":
                st.write(
                    "El cuadro clínico presenta alteraciones relevantes "
                    "que requieren correlacion con la evolución "
                    "del paciente.")
            else:
                st.write(
                    "El perfil clínico muestra alteraciones leves, "
                    "sin evidencia clara de compromiso severo.")

            # Alertas clinicas
            st.subheader("\u26a0\ufe0f Alertas clínicas")
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
                    "riesgo de cólico")

            if total_protein > 8.5:
                alertas.append(
                    "\U0001f7e1 Proteína elevada - "
                    "posible deshidratacion")

            if alertas:
                for alerta in alertas:
                    st.warning(alerta)
            else:
                st.success(
                    "\u2705 Sin alertas clínicas criticas")

            st.info(
                "\U0001f4a1 Revisa la pestana Historial "
                "para ver todos los caballos analizados")

        except Exception as e:
            st.error(f"Error en prediccion: {e}")

    st.divider()

st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.85em;">
        ⚠️ Herramienta de apoyo basada en análisis de variables clínicas. 
        Diseñada para complementar la evaluación veterinaria profesional, 
        no sustituye el juicio clínico. Desarrollado por Jorge Ojeda.
    </div>
    """,
    unsafe_allow_html=True
)
# ══════════════════════════════════════════════════════════
# TAB 2 - HISTORIAL
# ══════════════════════════════════════════════════════════
with tab2:

    st.header("\U0001f4cb Historial de caballos analizados")

    if not st.session_state.historial:
        st.info(
            "Aún no has analizado ningún caballo. "
            "Ve a la pestana Analizar caballo.")
    else:
        st.caption(
            f"Total analizados en esta sesión: "
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
