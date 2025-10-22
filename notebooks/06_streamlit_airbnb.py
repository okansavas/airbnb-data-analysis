# 06_streamlit_airbnb.py - DÜZELTİLMİŞ
# =====================================================
# 🏠 Airbnb Dashboard & Preisvorhersage
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Airbnb Dashboard", page_icon="🏠", layout="wide")

# ===============================
# 1️⃣ Titel & Einleitung
# ===============================
st.title("🏠 Airbnb Datenanalyse & Preisvorhersage")
st.write("Interaktives Dashboard zur Analyse von Airbnb-Daten in New York City")

# ===============================
# 2️⃣ Daten laden - KORRIGIERT
# ===============================
@st.cache_data
def load_data():
    try:
        # Aktuelles Verzeichnis (wo die Streamlit-App liegt)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Ein Verzeichnis hoch gehen zu airbnb-data-analysis
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data", "AB_NYC_2019.csv")
        
        st.write(f"📂 Lade Daten von: {data_path}")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.success(f"✅ Daten erfolgreich geladen: {df.shape} Zeilen x Spalten")
            return df
        else:
            st.error(f"❌ Datei nicht gefunden: {data_path}")
            # Fallback: Beispiel-Daten erstellen
            st.info("🔄 Erstelle Beispieldaten...")
            return create_sample_data()
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {e}")
        return create_sample_data()

def create_sample_data():
    """Fallback: Erstelle Beispieldaten wenn CSV nicht gefunden wird"""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'neighbourhood_group': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
        'price': np.random.randint(50, 300, n_samples),
        'minimum_nights': np.random.randint(1, 30, n_samples),
        'number_of_reviews': np.random.randint(0, 100, n_samples),
        'latitude': np.random.uniform(40.5, 40.9, n_samples),
        'longitude': np.random.uniform(-74.1, -73.7, n_samples)
    }
    return pd.DataFrame(data)

# Daten laden
df = load_data()

# ===============================
# 3️⃣ Seitenleiste (Filter)
# ===============================
st.sidebar.header("🔍 Filter")

if not df.empty:
    neigh_group = st.sidebar.multiselect(
        "Nachbarschaftsgruppe auswählen",
        options=df["neighbourhood_group"].unique(),
        default=df["neighbourhood_group"].unique()[:2]  # Nur erste 2 als Default
    )
    room_type = st.sidebar.multiselect(
        "Zimmertyp auswählen",
        options=df["room_type"].unique(),
        default=df["room_type"].unique()
    )
    price_limit = st.sidebar.slider("Maximaler Preis", 0, 500, 300)
    
    # Filter anwenden
    filtered_df = df[
        (df["neighbourhood_group"].isin(neigh_group)) &
        (df["room_type"].isin(room_type)) &
        (df["price"] <= price_limit)
    ]
else:
    st.warning("⚠️ Keine Daten verfügbar")
    filtered_df = pd.DataFrame()

# ===============================
# 4️⃣ KPI-Karten - KORRIGIERT
# ===============================
st.subheader("📊 Zusammenfassung")

if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Anzahl der Angebote", f"{filtered_df.shape[0]:,}")
    
    with col2:
        avg_price = filtered_df['price'].mean()
        st.metric("Durchschnittlicher Preis", f"${avg_price:.2f}")
    
    with col3:
        avg_nights = filtered_df['minimum_nights'].mean()
        st.metric("Durchschn. Mindestnächte", f"{avg_nights:.1f}")
    
    with col4:
        avg_reviews = filtered_df['number_of_reviews'].mean()
        st.metric("Durchschn. Bewertungen", f"{avg_reviews:.1f}")
else:
    st.warning("⚠️ Keine Daten nach Filterung")

# ===============================
# 5️⃣ Visualisierungen - KORRIGIERT
# ===============================
if not filtered_df.empty:
    # Preisverteilung
    st.subheader("💵 Preisverteilung nach Nachbarschaftsgruppe")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x="neighbourhood_group", y="price", ax=ax)
    plt.ylim(0, price_limit)
    plt.xticks(rotation=45)
    plt.title("Preisverteilung nach Nachbarschaftsgruppe")
    st.pyplot(fig)

    # Karte
    st.subheader("📍 Geografische Verteilung der Angebote")
    
    # Sicherstellen, dass Koordinaten vorhanden sind
    map_data = filtered_df[["latitude", "longitude"]].dropna()
    if not map_data.empty:
        st.map(map_data)
    else:
        st.warning("⚠️ Keine Koordinaten für Karte verfügbar")

    # Zusätzliche Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏘️ Verteilung nach Zimmertyp")
        room_counts = filtered_df['room_type'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        room_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        plt.title("Verteilung der Zimmertypen")
        st.pyplot(fig2)
    
    with col2:
        st.subheader("📈 Preis vs. Bewertungen")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        plt.scatter(filtered_df['number_of_reviews'], filtered_df['price'], alpha=0.5)
        plt.xlabel('Anzahl Bewertungen')
        plt.ylabel('Preis ($)')
        plt.title('Preis vs. Anzahl Bewertungen')
        st.pyplot(fig3)
else:
    st.info("ℹ️ Wählen Sie Filter aus, um Daten anzuzeigen")

# ===============================
# 6️⃣ Preisvorhersage - KORRIGIERT
# ===============================
st.sidebar.markdown("---")
st.sidebar.header("🤖 Preisvorhersage")

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, "models", "airbnb_model.pkl")
        encoders_path = os.path.join(project_root, "models", "label_encoders.pkl")
        
        st.sidebar.write(f"🔍 Suche Modelle in: {os.path.dirname(model_path)}")
        
        if os.path.exists(model_path) and os.path.exists(encoders_path):
            model = joblib.load(model_path)
            encoders = joblib.load(encoders_path)
            st.sidebar.success("✅ Modelle erfolgreich geladen")
            return model, encoders
        else:
            st.sidebar.warning(f"⚠️ Modelle nicht gefunden: {model_path}")
            return None, None
    except Exception as e:
        st.sidebar.error(f"❌ Fehler beim Laden der Modelle: {e}")
        return None, None

model, encoders = load_model()

if model is not None and encoders is not None:
    st.sidebar.success("✅ Preisvorhersage verfügbar")
    
    # Eingaben für Vorhersage
    n_group = st.sidebar.selectbox("Nachbarschaftsgruppe", df["neighbourhood_group"].unique())
    r_type = st.sidebar.selectbox("Zimmertyp", df["room_type"].unique())
    min_nights = st.sidebar.number_input("Minimale Nächte", min_value=1, max_value=365, value=3)
    n_reviews = st.sidebar.number_input("Anzahl der Bewertungen", min_value=0, value=10)
    
    # Standard-Koordinaten basierend auf Nachbarschaft
    default_coords = {
        'Manhattan': (40.78, -73.97),
        'Brooklyn': (40.65, -73.95),
        'Queens': (40.73, -73.79),
        'Bronx': (40.85, -73.87),
        'Staten Island': (40.58, -74.15)
    }
    
    default_lat, default_lon = default_coords.get(n_group, (40.7, -74.0))
    
    latitude = st.sidebar.number_input("Breitengrad", min_value=40.5, max_value=40.9, 
                                     value=float(default_lat), step=0.01, format="%.2f")
    longitude = st.sidebar.number_input("Längengrad", min_value=-74.1, max_value=-73.7, 
                                      value=float(default_lon), step=0.01, format="%.2f")

    if st.sidebar.button("💰 Preis Vorhersagen", type="primary"):
        try:
            # Encoding für kategorische Variablen
            neighbourhood_encoded = encoders['neighbourhood'].transform([n_group])[0]
            room_type_encoded = encoders['room_type'].transform([r_type])[0]
            
            # Eingabe-Daten vorbereiten
            X_input = np.array([[neighbourhood_encoded, room_type_encoded, latitude, 
                               longitude, min_nights, n_reviews]])
            
            # Vorhersage
            pred = model.predict(X_input)
            
            # Ergebnis anzeigen
            st.sidebar.success(f"💵 **Vorhergesagter Preis: ${pred[0]:.2f}**")
            
            # Zusätzliche Informationen
            with st.sidebar.expander("📋 Vorhersage-Details"):
                st.write(f"**Nachbarschaft:** {n_group}")
                st.write(f"**Zimmertyp:** {r_type}")
                st.write(f"**Mindestnächte:** {min_nights}")
                st.write(f"**Anzahl Bewertungen:** {n_reviews}")
                st.write(f"**Standort:** {latitude:.2f}, {longitude:.2f}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Fehler bei der Vorhersage: {str(e)}")
else:
    st.sidebar.warning("""
    ⚠️ **Preisvorhersage nicht verfügbar**
    
    Führen Sie zuerst diese Schritte durch:
    1. `02_preprocessing_airbnb.ipynb` ausführen
    2. `04_training_airbnb.ipynb` ausführen
    3. Streamlit App neu starten
    """)

# ===============================
# 7️⃣ Daten-Exploration - KORRIGIERT
# ===============================
st.subheader("📋 Daten-Exploration")

if not filtered_df.empty:
    tab1, tab2, tab3 = st.tabs(["📊 Statistiken", "🔍 Rohdaten", "📈 Charts"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Preis-Statistiken:**")
            st.write(filtered_df['price'].describe())
        
        with col2:
            st.write("**Mindestnächte-Statistiken:**")
            st.write(filtered_df['minimum_nights'].describe())
    
    with tab2:
        st.write(f"**Gefilterte Daten ({filtered_df.shape[0]} Zeilen):**")
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Verteilungen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preis-Histogramm
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            filtered_df['price'].hist(bins=30, ax=ax4)
            plt.title('Preis-Verteilung')
            plt.xlabel('Preis ($)')
            plt.ylabel('Häufigkeit')
            st.pyplot(fig4)
        
        with col2:
            # Nachbarschafts-Verteilung
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            filtered_df['neighbourhood_group'].value_counts().plot(kind='bar', ax=ax5)
            plt.title('Anzahl nach Nachbarschaft')
            plt.xticks(rotation=45)
            st.pyplot(fig5)
else:
    st.info("ℹ️ Wählen Sie Filter aus, um Daten anzuzeigen")

# ===============================
# 8️⃣ Footer
# ===============================
st.markdown("---")
st.caption("Erstellt von Okan • Powered by Streamlit & Pandas")

# Debug-Informationen (kann auskommentiert werden)
with st.expander("🔧 Debug-Informationen"):
    st.write("**Daten-Info:**")
    st.write(f"- Shape: {df.shape}")
    st.write(f"- Spalten: {list(df.columns)}")
    st.write(f"- Nachbarschaften: {list(df['neighbourhood_group'].unique())}")
    st.write(f"- Zimmertypen: {list(df['room_type'].unique())}")