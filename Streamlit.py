# Streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib.colors import rgb2hex

st.set_page_config(layout="wide")

DEFAULT_URL = "https://raw.githubusercontent.com/JotaBlanco/Pernas/refs/heads/main/comparativa_con_precios.csv"

st.sidebar.header("Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load default CSV if no file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_URL)

# Clean df
cols_two_decimals = [
    'Delta CanPres',
    'Castro €/ud', 'Castro € tot', 'Castro €/ud 2', 'Delta Castro',
    'Balboa €/ud', 'Balboa € tot', 'Balboa €/ud 2', 'Delta Balboa'
]
for col in cols_two_decimals:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

# Load capítulo/subcapítulo titles
TITULOS_URL = "https://raw.githubusercontent.com/JotaBlanco/Pernas/refs/heads/main/titulos_capitulos.csv"
df_titulos = pd.read_csv(TITULOS_URL)
df_titulos["Capitulo"] = df_titulos["Capitulo"].astype(str).str.zfill(2)
df_titulos["Subcapítulo"] = (df_titulos["Subcapítulo"].apply(lambda x: str(int(x)).zfill(2) if pd.notnull(x) else ""))

# Mapping for capítulos
capitulo_map = df_titulos[df_titulos['isCapitulo'] == True][['Capitulo', 'Resumen']].drop_duplicates()
capitulo_map = capitulo_map.set_index('Capitulo')['Resumen'].to_dict()

# Mapping for subcapítulos (Capitulo + Subcapítulo pair)
subcapitulo_map = df_titulos[df_titulos['isCapitulo'] == False][['Capitulo', 'Subcapítulo', 'Resumen']].drop_duplicates()


# -----------------------------
# 1️⃣ FULL-WIDTH DATAFRAME WITH CONDITIONAL FORMATTING
# -----------------------------
st.subheader("Visualizador bc3 (cambios subrayados)")

compare_pairs = [
    ("Ud", "Ud_2"),
    ("Resumen", "Resumen_2"),
    ("CanPres", "CanPres_2"),
    ("Pres", "Pres_2"),
    ("ImpPres", "ImpPres_2"),
    ("Descripción", "Descripción_2")
]

def style_differences(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of same shape with CSS styles.
    For any pair (col, col2) present in data, mark cells in `col2` where col != col2
    with a light blue background.
    """
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    for col, col2 in compare_pairs:
        if col in data.columns and col2 in data.columns:
            a = data[col]
            b = data[col2]
            neq_mask = ~(a.eq(b) | (a.isna() & b.isna()))
            # Changed to light blue with similar transparency as original red
            styles.loc[neq_mask, "Código"] = "background-color: rgba(0, 123, 255, 0.3);"
            styles.loc[neq_mask, col2] = "background-color: rgba(0, 123, 255, 0.3);"

    return styles

def get_differences_mask(data: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series indicating which rows have any differences between compared columns.
    """
    diff_mask = pd.Series(False, index=data.index)
    for col, col2 in compare_pairs:
        if col in data.columns and col2 in data.columns:
            a = data[col]
            b = data[col2]
            neq = ~(a.eq(b) | (a.isna() & b.isna()))
            diff_mask = diff_mask | neq
    return diff_mask

tab1, tab2 = st.tabs(["Todas las partidas", "Cambios"])

with tab1:
    styled_all = df.style.format(
        {col: "{:.2f}" for col in cols_two_decimals if col in df.columns}
    ).apply(style_differences, axis=None)
    st.dataframe(styled_all, use_container_width=True, height=900)

with tab2:
    diff_rows = df[get_differences_mask(df)]
    if diff_rows.empty:
        st.info("No hay diferencias en las columnas comparadas.")
    else:
        styled_diff = diff_rows.style.format(
            {col: "{:.2f}" for col in cols_two_decimals if col in diff_rows.columns}
        ).apply(style_differences, axis=None)
        st.dataframe(styled_diff, use_container_width=True, height=900)

# -----------------------------
# 2️⃣ AGGREGATED SUMS FOR EURO COLUMNS (con labels de CSV)
# -----------------------------
constructoras = ["Castro", "Balboa"]

# Paleta seaborn y mapa de colores consistente
palette = sns.color_palette("deep", n_colors=len(constructoras))
colores = {c: rgb2hex(palette[i]) for i, c in enumerate(constructoras)}

# Añadimos columna 'Capitulo' y 'Subcapitulo' en df principal con ceros a la izquierda
df["Capitulo"] = df["Código"].astype(str).str[:2].str.zfill(2)
df["Subcapitulo"] = df["Código"].astype(str).str.split(".").str[1].fillna("").str.zfill(2)

# -----------------------------
# Totales y Deltas por CAPÍTULO
# -----------------------------
agg_cap_totales = {
    c: df.groupby("Capitulo")[f"{c} € tot"].sum()
    if f"{c} € tot" in df.columns else pd.Series(dtype='float64')
    for c in constructoras
}
agg_cap_deltas = {
    c: df.groupby("Capitulo")[f"Delta {c}"].sum()
    if f"Delta {c}" in df.columns else pd.Series(dtype='float64')
    for c in constructoras
}

totales_cap_df = pd.concat([agg_cap_totales[c].rename(c) for c in constructoras], axis=1).fillna(0).reset_index()
deltas_cap_df = pd.concat([agg_cap_deltas[c].rename(c) for c in constructoras], axis=1).fillna(0).reset_index()

# Convertir también a str para merge
totales_cap_df["Capitulo"] = totales_cap_df["Capitulo"].astype(str).str.zfill(2)
deltas_cap_df["Capitulo"] = deltas_cap_df["Capitulo"].astype(str).str.zfill(2)

# Merge con labels de capítulos
df_labels_cap = df_titulos[df_titulos["isCapitulo"] == True][["Capitulo", "Resumen"]]
totales_cap_df = totales_cap_df.merge(df_labels_cap, on="Capitulo", how="left")
deltas_cap_df = deltas_cap_df.merge(df_labels_cap, on="Capitulo", how="left")

totales_cap_long = totales_cap_df.melt(
    id_vars=["Capitulo", "Resumen"], value_vars=constructoras,
    var_name="Constructora", value_name="Total €"
)
deltas_cap_long = deltas_cap_df.melt(
    id_vars=["Capitulo", "Resumen"], value_vars=constructoras,
    var_name="Constructora", value_name="Delta €"
)

orden_capitulos = df_labels_cap["Resumen"].tolist()
totales_cap_long["Resumen"] = pd.Categorical(totales_cap_long["Resumen"], categories=orden_capitulos, ordered=True)
deltas_cap_long["Resumen"] = pd.Categorical(deltas_cap_long["Resumen"], categories=orden_capitulos, ordered=True)

# -----------------------------
# 📊 GRÁFICOS - CAPÍTULO (col1 = deltas, col2 = totales)
# -----------------------------

st.markdown("---")
st.subheader("Totales y Deltas por Capítulo")

col1, col2 = st.columns([0.3, 0.7])

with col1:
    fig_deltas_cap = px.bar(
        deltas_cap_long,
        x="Delta €",
        y="Resumen",
        color="Constructora",
        color_discrete_map=colores,
        barmode="group",
        orientation="h",
        title="Deltas por Capítulo (€)",
        height=600
    )
    fig_deltas_cap.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray':orden_capitulos, 'showticklabels': False}
    )
    st.plotly_chart(fig_deltas_cap, use_container_width=True)

with col2:
    fig_totales_cap = px.bar(
        totales_cap_long,
        x="Total €",
        y="Resumen",  # etiqueta descriptiva
        color="Constructora",
        color_discrete_map=colores,
        barmode="group",
        orientation="h",
        title="Totales por Capítulo (€)",
        height=600
    )
    fig_totales_cap.update_layout(yaxis={'categoryorder':'array', 'categoryarray':orden_capitulos})
    st.plotly_chart(fig_totales_cap, use_container_width=True)

# -----------------------------
# Totales y Deltas por SUBCAPÍTULO (selector de Capítulo)
# -----------------------------

st.markdown("---")
st.subheader("Totales y Deltas por Subcapítulo")

# Selector de capítulo por etiqueta descriptiva
selected_capitulo_label = st.selectbox("Selecciona un Capítulo", options=orden_capitulos)
selected_capitulo_code = df_labels_cap[df_labels_cap["Resumen"] == selected_capitulo_label]["Capitulo"].iloc[0]
df_filtered = df[df["Capitulo"] == selected_capitulo_code]

agg_sub_totales = {
    c: df_filtered.groupby("Subcapitulo")[f"{c} € tot"].sum()
    if f"{c} € tot" in df_filtered.columns else pd.Series(dtype='float64')
    for c in constructoras
}
agg_sub_deltas = {
    c: df_filtered.groupby("Subcapitulo")[f"Delta {c}"].sum()
    if f"Delta {c}" in df_filtered.columns else pd.Series(dtype='float64')
    for c in constructoras
}

totales_sub_df = pd.concat([agg_sub_totales[c].rename(c) for c in constructoras], axis=1).fillna(0).reset_index()
deltas_sub_df = pd.concat([agg_sub_deltas[c].rename(c) for c in constructoras], axis=1).fillna(0).reset_index()

# Convertir Subcapitulo a str con ceros a la izquierda
totales_sub_df["Subcapitulo"] = totales_sub_df["Subcapitulo"].astype(str).str.zfill(2)
deltas_sub_df["Subcapitulo"] = deltas_sub_df["Subcapitulo"].astype(str).str.zfill(2)

# Merge con labels de subcapítulos (filtrados por capítulo)
df_labels_sub = df_titulos[
    (df_titulos["isCapitulo"] == False) &
    (df_titulos["Capitulo"] == selected_capitulo_code)
][["Capitulo", "Subcapítulo", "Resumen"]]

totales_sub_df = totales_sub_df.merge(df_labels_sub, left_on="Subcapitulo", right_on="Subcapítulo", how="left")
deltas_sub_df = deltas_sub_df.merge(df_labels_sub, left_on="Subcapitulo", right_on="Subcapítulo", how="left")

# Añadir columna "Capitulo" para filtro en largos
totales_sub_df["Capitulo"] = selected_capitulo_code
deltas_sub_df["Capitulo"] = selected_capitulo_code

totales_sub_long = totales_sub_df.melt(
    id_vars=["Capitulo", "Subcapitulo", "Resumen"], value_vars=constructoras,
    var_name="Constructora", value_name="Total €"
)
deltas_sub_long = deltas_sub_df.melt(
    id_vars=["Capitulo", "Subcapitulo", "Resumen"], value_vars=constructoras,
    var_name="Constructora", value_name="Delta €"
)

orden_subcap = df_labels_sub["Resumen"].tolist()
totales_sub_long["Resumen"] = pd.Categorical(totales_sub_long["Resumen"], categories=orden_subcap, ordered=True)
deltas_sub_long["Resumen"] = pd.Categorical(deltas_sub_long["Resumen"], categories=orden_subcap, ordered=True)

# -----------------------------
# 📊 GRÁFICOS - SUBCAPÍTULO (col1 = deltas, col2 = totales)
# -----------------------------

col1, col2 = st.columns([0.3, 0.7])

totales_sub_sel = totales_sub_long[totales_sub_long['Capitulo'] == selected_capitulo_code]
deltas_sub_sel = deltas_sub_long[deltas_sub_long['Capitulo'] == selected_capitulo_code]

with col1:
    fig_deltas_sub = px.bar(
        deltas_sub_sel,
        x="Delta €",
        y="Resumen",
        color="Constructora",
        color_discrete_map=colores,
        barmode="group",
        orientation="h",
        title=f"Deltas por Subcapítulo de {selected_capitulo_label} (€)",
        height=600
    )
    fig_deltas_sub.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray':orden_subcap, 'showticklabels': False}
    )
    st.plotly_chart(fig_deltas_sub, use_container_width=True)

with col2:
    fig_totales_sub = px.bar(
        totales_sub_sel,
        x="Total €",
        y="Resumen",
        color="Constructora",
        color_discrete_map=colores,
        barmode="group",
        orientation="h",
        title=f"Totales por Subcapítulo de {selected_capitulo_label} (€)",
        height=600
    )
    fig_totales_sub.update_layout(yaxis={'categoryorder':'array', 'categoryarray':orden_subcap})
    st.plotly_chart(fig_totales_sub, use_container_width=True)
