# Streamlit.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.sidebar.header("Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # 1️⃣ FULL-WIDTH DATAFRAME WITH CONDITIONAL FORMATTING
    # -----------------------------
    st.subheader("DataFrame View (Differences Highlighted)")

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
        For any pair (col, col2) present in data, mark cells in `col` where col != col2
        with a light red background with ~50% opacity.
        """
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for col, col2 in compare_pairs:
            if col in data.columns and col2 in data.columns:
                # Compare safely handling NaNs: different if one is NaN and other not, or values unequal
                a = data[col]
                b = data[col2]

                # mask True where they differ (including NaN vs value)
                neq_mask = ~(a.eq(b) | (a.isna() & b.isna()))
                # apply style to the -2 column cells
                styles.loc[neq_mask, "Código"] = "background-color: rgba(255, 0, 0, 0.5);"
                styles.loc[neq_mask, col2] = "background-color: rgba(255, 0, 0, 0.5);"

        return styles

    # Apply the styler using axis=None (styler receives full dataframe of styles)
    styled = df.style.apply(style_differences, axis=None)

    # Show full-width dataframe
    st.dataframe(styled, use_container_width=True, height=900)

    # -----------------------------
    # 2️⃣ AGGREGATED SUMS FOR EURO COLUMNS
    # -----------------------------
    st.subheader("Aggregated € Columns")

    euro_cols = [c for c in df.columns if '€ tot' in c]
    if 'Código' in df.columns and euro_cols:
        # create grouping keys
        parts = df['Código'].astype(str).str.split('.', expand=True)
        df['XX'] = parts[0].fillna('')
        df['XX_YY'] = parts[0].fillna('') + '.' + parts[1].fillna('')

        # convert euro cols to numeric (coerce errors) before summing
        df_euros = df.copy()
        for c in euro_cols:
            df_euros[c] = pd.to_numeric(df_euros[c].replace('[€,]', '', regex=True), errors='coerce')

        agg_xx = df_euros.groupby('XX')[euro_cols].sum(numeric_only=True).reset_index()
        agg_xxyy = df_euros.groupby('XX_YY')[euro_cols].sum(numeric_only=True).reset_index()

        # Convert € columns to int
        agg_xx[euro_cols] = agg_xx[euro_cols].astype(int)
        agg_xxyy[euro_cols] = agg_xxyy[euro_cols].astype(int)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Grouped by **XX**:")
            st.dataframe(agg_xx, use_container_width=True)
        with col2:
            st.write("Grouped by **XX.YY**:")
            st.dataframe(agg_xxyy, use_container_width=True)
    else:
        st.write("No `Código` column found or no columns containing '€' in their name.")

    # -----------------------------
    # 3️⃣ COLUMN DISTRIBUTION
    # -----------------------------
    st.subheader("Column Distribution")
    selected_col = st.selectbox("Select a column", df.columns)

    if pd.api.types.is_numeric_dtype(df[selected_col]):
        fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
    else:
        counts = df[selected_col].astype(str).value_counts().reset_index()
        counts.columns = [selected_col, 'count']
        fig = px.bar(counts, x=selected_col, y='count', title=f"Distribution of {selected_col}")
    st.plotly_chart(fig, use_container_width=True)