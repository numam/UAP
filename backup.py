import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import re
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Penampakan UFO",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create models directory if not exists
MODELS_DIR = "saved_models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛸 Analisis Penampakan UFO untuk Strategi Konten/Film</h1>', unsafe_allow_html=True)

# Helper Functions - MUST BE DEFINED BEFORE SIDEBAR
def save_model(model, model_name, label_encoder, scaler, history, metadata=None, extra_encoders=None):
    """Save model, encoders, extra encoders, and metadata to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(MODELS_DIR, f"{safe_name}_{timestamp}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    
    # Save encoders and scaler
    with open(os.path.join(save_path, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(save_path, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save any extra encoders provided (e.g., le_city, le_state, le_country)
    if isinstance(extra_encoders, dict):
        for name, enc in extra_encoders.items():
            try:
                with open(os.path.join(save_path, f"{name}.pkl"), 'wb') as ef:
                    pickle.dump(enc, ef)
            except Exception:
                continue
    
    # Save history
    with open(os.path.join(save_path, "history.json"), 'w') as f:
        json.dump(history, f)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_name': model_name,
        'timestamp': timestamp,
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist()
    })
    
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    return save_path

def load_saved_models():
    """Load all saved models info"""
    saved_models = []
    
    if not os.path.exists(MODELS_DIR):
        return saved_models
    
    for model_dir in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir)
        metadata_path = os.path.join(model_path, "metadata.json")
        
        if os.path.isdir(model_path) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                saved_models.append({
                    'path': model_path,
                    'name': metadata.get('model_name', 'Unknown'),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'num_classes': metadata.get('num_classes', 0),
                    'metadata': metadata
                })
            except Exception as e:
                continue
    
    return sorted(saved_models, key=lambda x: x['timestamp'], reverse=True)

def load_model_from_disk(model_path, input_size):
    """Load model from disk"""
    # Load metadata
    with open(os.path.join(model_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    num_classes = metadata['num_classes']
    
    # Initialize model architecture
    if "MLP" in model_name:
        model = MLPClassifier(input_size, num_classes)
    elif "TabNet" in model_name:
        model = SimpleTabNet(input_size, num_classes)
    else:  # FT-Transformer
        model = FTTransformer(input_size, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), 
                                     map_location=torch.device('cpu')))
    
    # Load encoders
    with open(os.path.join(model_path, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(model_path, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load history
    with open(os.path.join(model_path, "history.json"), 'r') as f:
        history = json.load(f)
    
    return model, label_encoder, scaler, history, metadata

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ufo.png", width=100)
    st.markdown("### 📊 Menu Navigasi")
    page = st.radio("", ["🏠 Home", "🔍 Analisis Query", "📍 Analisis Lokasi", 
                         "🤖 Model Training", "📈 Evaluasi Model", "💾 Model Management"])
    
    st.markdown("---")
    st.markdown("### 📁 Dataset")
    
    # Check if dataset exists (use robust path relative to this file)
    BASE_DIR = os.path.dirname(__file__)
    dataset_path = os.path.join(BASE_DIR, "ufo_sightings.csv")

    if os.path.exists(dataset_path):
        st.success("✅ Dataset loaded")
        st.info(f"📄 `{dataset_path}`")
    else:
        st.error("❌ Dataset tidak ditemukan")
        st.warning("Silakan letakkan file `ufo_sightings.csv` di folder yang sama dengan `app.py`")
    
    st.markdown("---")
    st.markdown("### 💾 Saved Models")
    
    saved_models = load_saved_models()
    if saved_models:
        st.success(f"✅ {len(saved_models)} model(s) saved")
        for i, model_info in enumerate(saved_models[:3], 1):
            st.caption(f"{i}. {model_info['name']}")
    else:
        st.info("No saved models yet")

# Load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Clean column names
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_data(df):
    # Select relevant columns
    cols = ['Location.City', 'Location.State', 'Location.Country', 
            'Data.Shape', 'Data.Description excerpt', 'Dates.Sighted.Year']
    
    # Handle different column name formats
    for col in cols:
        if col not in df.columns:
            # Try alternative formats
            alt_col = col.replace('.', ' ')
            if alt_col in df.columns:
                df[col] = df[alt_col]
    
    df_clean = df[cols].copy()
    df_clean = df_clean.dropna(subset=['Data.Shape'])
    
    # Clean text data
    df_clean['Location.City'] = df_clean['Location.City'].str.lower().str.strip()
    df_clean['Location.State'] = df_clean['Location.State'].str.upper().str.strip()
    df_clean['Data.Shape'] = df_clean['Data.Shape'].str.lower().str.strip()
    
    return df_clean
    """Save model, encoders, and metadata to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(MODELS_DIR, f"{safe_name}_{timestamp}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    
    # Save encoders and scaler
    with open(os.path.join(save_path, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(save_path, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save history
    with open(os.path.join(save_path, "history.json"), 'w') as f:
        json.dump(history, f)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_name': model_name,
        'timestamp': timestamp,
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist()
    })
    
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    return save_path

def load_saved_models():
    """Load all saved models info"""
    saved_models = []
    
    if not os.path.exists(MODELS_DIR):
        return saved_models
    
    for model_dir in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir)
        metadata_path = os.path.join(model_path, "metadata.json")
        
        if os.path.isdir(model_path) and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            saved_models.append({
                'path': model_path,
                'name': metadata.get('model_name', 'Unknown'),
                'timestamp': metadata.get('timestamp', 'Unknown'),
                'num_classes': metadata.get('num_classes', 0),
                'metadata': metadata
            })
    
    return sorted(saved_models, key=lambda x: x['timestamp'], reverse=True)

def load_model_from_disk(model_path, input_size):
    """Load model from disk"""
    # Load metadata
    with open(os.path.join(model_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata['model_name']
    num_classes = metadata['num_classes']
    
    # Initialize model architecture
    if "MLP" in model_name:
        model = MLPClassifier(input_size, num_classes)
    elif "TabNet" in model_name:
        model = SimpleTabNet(input_size, num_classes)
    else:  # FT-Transformer
        model = FTTransformer(input_size, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), 
                                     map_location=torch.device('cpu')))
    
    # Load encoders
    with open(os.path.join(model_path, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(model_path, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load history
    with open(os.path.join(model_path, "history.json"), 'r') as f:
        history = json.load(f)
    
    return model, label_encoder, scaler, history, metadata

def extract_location_from_query(query, df):
    """Extract city and state from natural language query"""
    query_lower = query.lower()

    city = None
    state = None

    # 1) Prefer explicit phrase patterns like 'latar <city> <STATE>' or 'di <city> <STATE>'
    m = re.search(r"(?:latar|di|in|at)\s+([a-zA-Z\s]+?)\s+([A-Za-z]{2})\b", query, re.IGNORECASE)
    if m:
        city = m.group(1).strip().lower()
        state = m.group(2).upper()
        # verify city exists
        if city not in df['Location.City'].str.lower().values:
            # try to trim trailing words (e.g., 'anchorage?')
            city = city.split()[-1]
            if city not in df['Location.City'].str.lower().values:
                city = None

    # 2) Look for explicit two-letter state codes anywhere (AK, CA, etc.)
    if not state:
        state_matches = re.findall(r"\b([A-Za-z]{2})\b", query)
        for s in state_matches:
            su = s.upper()
            if su in df['Location.State'].unique():
                state = su
                break

    # 3) Find city by checking known cities (longest first to avoid partial matches like 'film')
    if not city:
        cities = sorted(df['Location.City'].dropna().unique(), key=lambda x: len(str(x)), reverse=True)
        for known_city in cities:
            kc = str(known_city).lower()
            # match whole word boundaries to avoid false positives
            if re.search(r"\b" + re.escape(kc) + r"\b", query_lower):
                city = kc
                break

    # 4) If city found but state still missing, try to infer state from dataset for that city
    if city and not state:
        matches = df[df['Location.City'].str.lower() == city]['Location.State'].unique()
        if len(matches) == 1:
            state = matches[0]

    # 5) If still nothing, try to find any known state name or code in the query
    if not state:
        for known_state in df['Location.State'].dropna().unique():
            if re.search(r"\b" + re.escape(str(known_state)) + r"\b", query.upper()):
                state = known_state
                break

    return city, state

def generate_recommendation_text(filtered_df, location_name, top_shape):
    """Generate natural language recommendation"""
    shape_counts = filtered_df['Data.Shape'].value_counts()
    total = len(filtered_df)
    top_count = shape_counts.values[0]
    top_percentage = (top_count / total) * 100
    
    # Extract characteristics
    top_shape_data = filtered_df[filtered_df['Data.Shape'] == top_shape]
    descriptions = ' '.join(top_shape_data['Data.Description excerpt'].astype(str)).lower()
    
    # Detect characteristics
    characteristics = []
    
    # Quantity
    if any(word in descriptions for word in ['multiple', 'many', 'several', 'numerous', 'group', 'three', 'four', 'five']):
        characteristics.append("sering muncul dalam jumlah banyak sekaligus")
    
    # Colors
    colors = ['orange', 'red', 'blue', 'white', 'green', 'yellow', 'bright']
    found_colors = [c for c in colors if c in descriptions]
    if found_colors:
        color_str = ", ".join(found_colors[:3])
        characteristics.append(f"berwarna {color_str}")
    
    # Size
    if any(word in descriptions for word in ['large', 'huge', 'big', 'enormous']):
        characteristics.append("berukuran besar")
    elif any(word in descriptions for word in ['small', 'tiny', 'little']):
        characteristics.append("berukuran kecil")
    
    # Brightness
    if 'bright' in descriptions or 'glowing' in descriptions:
        characteristics.append("memancarkan cahaya terang")
    
    # Movement
    if 'fast' in descriptions or 'rapid' in descriptions:
        characteristics.append("bergerak cepat")
    elif 'hovering' in descriptions or 'stationary' in descriptions:
        characteristics.append("melayang atau diam di tempat")
    elif 'slow' in descriptions:
        characteristics.append("bergerak lambat")
    
    # Pattern
    if 'formation' in descriptions or 'pattern' in descriptions:
        characteristics.append("membentuk formasi tertentu")
    
    # Build recommendation text
    char_text = ", ".join(characteristics) if characteristics else "memiliki karakteristik yang bervariasi"
    
    # Get most common years
    common_years = top_shape_data['Dates.Sighted.Year'].value_counts().head(3)
    if len(common_years) > 0:
        years_text = ", ".join([str(year) for year in common_years.index])
        year_info = f" Bentuk ini paling sering terlihat pada tahun {years_text}."
    else:
        year_info = ""
    
    recommendation = f"""
Bentuk **{top_shape.upper()}** sangat cocok untuk membuat film atau konten tentang UFO dengan latar {location_name}. 

Bentuk ini dilaporkan **paling sering muncul** di lokasi tersebut dengan total **{top_count} penampakan ({top_percentage:.1f}%)** dari seluruh kasus yang tercatat. 

**Ciri-ciri khas:** {char_text}.{year_info}

Dengan karakteristik yang konsisten ini, bentuk {top_shape} akan memberikan kredibilitas dan autentisitas pada cerita film Anda, karena sesuai dengan pola penampakan yang sebenarnya terjadi di lokasi tersebut.
"""
    
    return recommendation.strip()
    """Extract common features from descriptions for each shape"""
    shape_features = {}
    
    for shape in shapes.unique():
        shape_desc = descriptions[shapes == shape]
        all_words = ' '.join(shape_desc.astype(str)).lower()
        
        # Common descriptive words
        colors = ['red', 'orange', 'blue', 'green', 'white', 'yellow', 'bright']
        quantities = ['multiple', 'many', 'several', 'numerous', 'group']
        sizes = ['large', 'small', 'huge', 'tiny', 'big']
        movements = ['fast', 'slow', 'hovering', 'moving', 'flying']
        
        features = {
            'colors': [c for c in colors if c in all_words],
            'quantities': [q for q in quantities if q in all_words],
            'sizes': [s for s in sizes if s in all_words],
            'movements': [m for m in movements if m in all_words]
        }
        
        shape_features[shape] = features
    
    return shape_features

# Neural Network Models
class UFODataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    """Feedforward Neural Network (MLP)"""
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64, 32]):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  # Changed from BatchNorm to LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SimpleTabNet(nn.Module):
    """Simplified TabNet-inspired architecture"""
    def __init__(self, input_size, num_classes):
        super(SimpleTabNet, self).__init__()
        
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm to LayerNorm
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Attention produces a weight per feature (same dim as features)
        self.attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_transformer(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        return self.classifier(attended_features)

class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer"""
    def __init__(self, input_size, num_classes, d_model=64, nhead=4, num_layers=2):
        super(FTTransformer, self).__init__()
        
        self.feature_embedding = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    metric_cols = st.columns(4)
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - Best: {best_val_acc:.2f}%')
        
        # Update metrics
        with metric_cols[0]:
            st.metric("Train Loss", f"{train_loss:.4f}")
        with metric_cols[1]:
            st.metric("Train Acc", f"{train_acc:.2f}%")
        with metric_cols[2]:
            st.metric("Val Loss", f"{val_loss:.4f}")
        with metric_cols[3]:
            st.metric("Val Acc", f"{val_acc:.2f}%")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            st.info(f"⏹️ Early stopping at epoch {epoch+1}. No improvement for {early_stop_patience} epochs.")
            break
    
    return history

def evaluate_model(model, test_loader, label_encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return all_preds, all_labels

# Main App Logic
# `dataset_path` is defined in the sidebar section above using a file-relative path

if os.path.exists(dataset_path):
    df = load_data(dataset_path)
    df_clean = preprocess_data(df)
    
    if page == "🏠 Home":
        st.markdown('<p class="sub-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_clean):,}")
        with col2:
            st.metric("Unique Shapes", len(df_clean['Data.Shape'].unique()))
        with col3:
            st.metric("Countries", len(df_clean['Location.Country'].unique()))
        with col4:
            st.metric("Years Range", f"{df_clean['Dates.Sighted.Year'].min()}-{df_clean['Dates.Sighted.Year'].max()}")
        
        st.markdown("### 📋 Sample Data")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Shape distribution
        st.markdown("### 📊 Distribusi Bentuk UFO")
        shape_counts = df_clean['Data.Shape'].value_counts()
        
        fig = px.bar(x=shape_counts.index, y=shape_counts.values,
                    labels={'x': 'Shape', 'y': 'Count'},
                    title='Top UFO Shapes',
                    color=shape_counts.values,
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Yearly trend
        st.markdown("### 📈 Tren Penampakan per Tahun")
        yearly = df_clean.groupby('Dates.Sighted.Year').size()
        fig2 = px.line(x=yearly.index, y=yearly.values,
                      labels={'x': 'Year', 'y': 'Sightings'},
                      title='UFO Sightings Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    
    elif page == "🔍 Analisis Query":
        st.markdown('<p class="sub-header">🔍 Pencarian Cerdas dengan Natural Language</p>', unsafe_allow_html=True)
        
        st.info("💡 **Tip:** Tanyakan dalam bahasa natural! Contoh: 'bentuk ufo apa yang cocok untuk film di anchorage AK?'")
        
        # Examples
        with st.expander("📝 Lihat Contoh Query"):
            st.markdown("""
            **Contoh pertanyaan yang bisa Anda ajukan:**
            - `bentuk ufo apa yang cocok jika membuat film tentang ufo dengan latar anchorage AK?`
            - `apa bentuk UFO paling umum di seattle WA?`
            - `rekomendasi bentuk UFO untuk konten tentang penampakan di portland`
            - `bentuk apa yang sering muncul di california?`
            - `what UFO shape is most common in new york NY?`
            """)
        
        # Single natural language input
        user_query = st.text_area(
            "🎤 Masukkan pertanyaan Anda:", 
            placeholder="Contoh: bentuk ufo apa yang cocok jika membuat film tentang ufo dengan latar anchorage AK?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Analisis", type="primary", use_container_width=True)
        with col2:
            if user_query:
                st.caption(f"📊 Query length: {len(user_query)} characters")
        
        if search_button and user_query:
            with st.spinner("🔍 Menganalisis query dan mencari data..."):
                # If query asks for locations suitable for a given shape, handle that first
                q_lower = user_query.lower()
                handled = False
                city = None
                state = None
                shapes_list = df_clean['Data.Shape'].dropna().unique().astype(str)
                found_shape = None
                for sh in sorted(shapes_list, key=lambda x: len(x), reverse=True):
                    if re.search(r"\b" + re.escape(sh.lower()) + r"\b", q_lower):
                        found_shape = sh
                        break

                location_query_patterns = [
                    r'latar kota mana', r'latar mana', r'lokasi .* cocok', r'kota .* cocok', r'lokasi yang cocok',
                    r'which city.*for', r'best city.*for', r'kota yang cocok untuk', r'latar .* cocok untuk'
                ]

                is_shape_location_query = False
                if found_shape:
                    for p in location_query_patterns:
                        if re.search(p, q_lower):
                            is_shape_location_query = True
                            break
                
                if is_shape_location_query and found_shape:
                    shape = found_shape
                    # compute per-city suitability: proportion of sightings in the city that are this shape
                    gb = df_clean.groupby(['Location.City', 'Location.State'])['Data.Shape'].agg(list).reset_index()
                    city_stats = []
                    for _, row in gb.iterrows():
                        lst = [s for s in row['Data.Shape'] if isinstance(s, str)]
                        total = len(lst)
                        count_shape = sum(1 for s in lst if s.lower() == shape.lower())
                        if total >= 5 and count_shape > 0:
                            pct = (count_shape / total) * 100
                            city_stats.append((row['Location.City'], row['Location.State'], count_shape, total, pct))

                    if not city_stats:
                        st.info(f"⚠️ Tidak ditemukan kota dengan cukup data untuk bentuk '{shape}'")
                    else:
                        # sort by percentage then by absolute count
                        # Prefer higher absolute counts first, then higher percentage
                        city_stats.sort(key=lambda x: (x[2], x[4]), reverse=True)
                        top_city, top_state, top_count, top_total, top_pct = city_stats[0]
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>📍 Rekomendasi Lokasi untuk Bentuk {shape.upper()}</h3>
                            <p><strong>{top_city.title()}, {top_state}</strong> cocok untuk bentuk {shape} — {top_count} penampakan ({top_pct:.1f}% dari {top_total} penampakan di kota ini).</p>
                        </div>
                        """, unsafe_allow_html=True)
                    # mark handled so the normal location-based flow is skipped
                    handled = True
                # Normal location-based flow (only if not handled above)
                if not handled:
                    city, state = extract_location_from_query(user_query, df_clean)

                    if not city and not state:
                        st.error("❌ Tidak dapat mendeteksi lokasi dari query Anda. Pastikan Anda menyebutkan nama kota atau state.")
                        st.info("💡 Coba tambahkan nama kota atau kode state (contoh: 'di anchorage AK' atau 'di california')")
                    else:
                        # Show detected location
                        location_parts = []
                        if city:
                            location_parts.append(f"Kota: **{city.title()}**")
                        if state:
                            location_parts.append(f"State: **{state}**")
                        
                        st.success(f"✅ Lokasi terdeteksi → {' | '.join(location_parts)}")
                        
                        # Filter data
                        if city and state:
                            filtered_df = df_clean[
                                (df_clean['Location.City'] == city) & 
                                (df_clean['Location.State'] == state)
                            ]
                            location_name = f"{city.title()}, {state}"
                        elif city:
                            filtered_df = df_clean[df_clean['Location.City'] == city]
                            location_name = city.title()
                        else:  # state only
                            filtered_df = df_clean[df_clean['Location.State'] == state]
                            location_name = state
                    
                    if len(filtered_df) > 0:
                        st.markdown("---")
                        
                        # Shape analysis
                        shape_counts = filtered_df['Data.Shape'].value_counts()
                        total_sightings = len(filtered_df)
                        
                        # Get top shape
                        top_shape = shape_counts.index[0]
                        
                        # Generate recommendation
                        recommendation = generate_recommendation_text(filtered_df, location_name, top_shape)
                        
                        # Display main recommendation
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>🎬 Rekomendasi untuk Film/Konten UFO</h3>
                            {recommendation}
                        </div>
                        """, unsafe_allow_html=True)

                        # --- ML-based prediction (Option A, prefer session models) ---
                        try:
                            model_ml = None
                            le_shape_ml = None
                            scaler_ml = None
                            model_source = None

                            # Prefer models loaded in session (prefer TabNet if available)
                            if 'models' in st.session_state and len(st.session_state.models) > 0:
                                session_items = list(st.session_state.models.items())
                                # Prefer TabNet models in session
                                tabnet_items = [it for it in session_items if 'tabnet' in it[0].lower()]
                                if tabnet_items:
                                    best_name, best_data = max(
                                        tabnet_items,
                                        key=lambda item: max(item[1]['history'].get('val_acc', [0]))
                                    )
                                else:
                                    best_name, best_data = max(
                                        session_items,
                                        key=lambda item: max(item[1]['history'].get('val_acc', [0]))
                                    )

                                model_ml = best_data['model']
                                le_shape_ml = best_data['label_encoder']
                                scaler_ml = best_data['scaler']
                                model_source = f"session:{best_name}"

                            else:
                                # Fallback to best saved model on disk
                                saved = load_saved_models()
                                if saved:
                                    # Prefer TabNet saved models if any
                                    tabnet_saved = [m for m in saved if 'tabnet' in m.get('name', '').lower() or 'tabnet' in m.get('metadata', {}).get('model_name', '').lower()]
                                    candidates = tabnet_saved if tabnet_saved else saved
                                    best_info = max(candidates, key=lambda m: m['metadata'].get('best_val_acc', 0))
                                    try:
                                        model_ml, le_shape_ml, scaler_ml, _, _ = load_model_from_disk(best_info['path'], input_size=4)
                                        model_source = f"disk:{best_info['name']}"
                                    except Exception:
                                        model_ml = None

                            if model_ml is not None:
                                with st.expander(f"🔮 ML Prediction ({model_source})", expanded=False):
                                    # Prepare encoders for city/state/country using same filtering as training
                                    df_model_all = df_clean.copy()
                                    shape_counts_all = df_model_all['Data.Shape'].value_counts()
                                    valid_shapes_all = shape_counts_all[shape_counts_all >= 10].index
                                    df_model_all = df_model_all[df_model_all['Data.Shape'].isin(valid_shapes_all)]

                                    le_city = None
                                    le_state = None
                                    le_country = None

                                    # Try to load encoders saved with model (if disk model used)
                                    if model_source and model_source.startswith('disk:'):
                                        model_dir = best_info['path'] if 'best_info' in locals() else None
                                        if model_dir and os.path.exists(model_dir):
                                            try:
                                                p_city = os.path.join(model_dir, 'le_city.pkl')
                                                p_state = os.path.join(model_dir, 'le_state.pkl')
                                                p_country = os.path.join(model_dir, 'le_country.pkl')
                                                if os.path.exists(p_city):
                                                    with open(p_city, 'rb') as fenc:
                                                        le_city = pickle.load(fenc)
                                                if os.path.exists(p_state):
                                                    with open(p_state, 'rb') as fenc:
                                                        le_state = pickle.load(fenc)
                                                if os.path.exists(p_country):
                                                    with open(p_country, 'rb') as fenc:
                                                        le_country = pickle.load(fenc)
                                            except Exception:
                                                le_city = le_state = le_country = None

                                    # If any encoder missing, fit from dataset (fallback)
                                    if len(df_model_all) == 0:
                                        st.warning("⚠️ Tidak ada data yang cukup untuk membuat encoder kota/state/country untuk prediksi ML")
                                    else:
                                        if le_city is None or le_state is None or le_country is None:
                                            le_city = LabelEncoder()
                                            le_state = LabelEncoder()
                                            le_country = LabelEncoder()
                                            le_city.fit(df_model_all['Location.City'].astype(str))
                                            le_state.fit(df_model_all['Location.State'].astype(str))
                                            le_country.fit(df_model_all['Location.Country'].astype(str))

                                        # Representative year
                                        if not filtered_df['Dates.Sighted.Year'].mode().empty:
                                            pred_year = int(filtered_df['Dates.Sighted.Year'].mode().iloc[0])
                                        else:
                                            pred_year = int(filtered_df['Dates.Sighted.Year'].median())

                                        city_val = city if city else (df_model_all['Location.City'].mode().iloc[0] if not df_model_all['Location.City'].mode().empty else df_model_all['Location.City'].iloc[0])
                                        state_val = state if state else (df_model_all['Location.State'].mode().iloc[0] if not df_model_all['Location.State'].mode().empty else df_model_all['Location.State'].iloc[0])

                                        if city_val in df_model_all['Location.City'].values:
                                            country_candidates = df_model_all[df_model_all['Location.City'] == city_val]['Location.Country']
                                            country_val = country_candidates.mode().iloc[0] if not country_candidates.mode().empty else df_model_all['Location.Country'].mode().iloc[0]
                                        else:
                                            country_val = df_model_all['Location.Country'].mode().iloc[0]

                                        def safe_encode(enc, v):
                                            v = str(v)
                                            if v in enc.classes_:
                                                return int(enc.transform([v])[0])
                                            return 0

                                        city_enc = safe_encode(le_city, city_val)
                                        state_enc = safe_encode(le_state, state_val)
                                        country_enc = safe_encode(le_country, country_val)

                                        X_pred = np.array([[city_enc, state_enc, country_enc, pred_year]])
                                        try:
                                            X_pred_scaled = scaler_ml.transform(X_pred)
                                        except Exception:
                                            # If scaler incompatible, skip ML prediction
                                            st.warning("⚠️ Scaler model tidak cocok dengan input; ML prediction dibatalkan.")
                                            X_pred_scaled = None

                                        if X_pred_scaled is not None:
                                            model_ml.eval()
                                            with torch.no_grad():
                                                X_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32)
                                                outputs = model_ml(X_tensor)
                                                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                                            topk_idx = probs.argsort()[::-1][:3]
                                            try:
                                                top_shapes = le_shape_ml.inverse_transform(topk_idx)
                                            except Exception:
                                                # fallback: use indices as labels
                                                top_shapes = [str(i) for i in topk_idx]
                                            top_probs = (probs[topk_idx] * 100).round(2)

                                            pred_lines = ""
                                            for i, (s, p) in enumerate(zip(top_shapes, top_probs), 1):
                                                pred_lines += f"<p><strong>#{i} {s.upper()}</strong> — {p}%</p>"

                                            st.markdown(f"""
                                            <div style='background:linear-gradient(90deg,#4b2f88,#667eea); padding:12px; border-radius:8px; color:white;'>
                                                <h4 style='margin:0 0 6px 0;'>🔮 Prediksi Model: Bentuk Teratas untuk {location_name}</h4>
                                                {pred_lines}
                                                <p style='opacity:0.9; margin-top:6px; font-size:0.9em;'>Sumber model: {model_source}</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                            else:
                                st.info("ℹ️ Tidak ada model yang tersedia untuk prediksi ML")
                        except Exception as e:
                            st.warning(f"⚠️ ML prediction gagal: {e}")
                        
                        # Visual statistics
                        st.markdown("### 📊 Statistik Detail")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Penampakan", f"{total_sightings:,}")
                        with col2:
                            st.metric("Bentuk Teratas", top_shape.upper())
                        with col3:
                            top_percentage = (shape_counts.values[0] / total_sightings) * 100
                            st.metric("Dominasi", f"{top_percentage:.1f}%")
                        
                        # Other shapes ranking
                        st.markdown("### 🏅 Peringkat Bentuk UFO Lainnya")
                        
                        ranking_data = []
                        for i, (shape, count) in enumerate(shape_counts.items(), 1):
                            percentage = (count / total_sightings) * 100
                            
                            # Get features for this shape
                            shape_data = filtered_df[filtered_df['Data.Shape'] == shape]
                            shape_desc = ' '.join(shape_data['Data.Description excerpt'].astype(str)).lower()
                            
                            features = []
                            if 'bright' in shape_desc or 'glowing' in shape_desc:
                                features.append("cahaya terang")
                            if 'multiple' in shape_desc or 'many' in shape_desc:
                                features.append("jumlah banyak")
                            
                            colors = [c for c in ['orange', 'red', 'blue', 'white', 'green'] if c in shape_desc]
                            if colors:
                                features.append(f"warna {colors[0]}")
                            
                            if 'fast' in shape_desc:
                                features.append("gerak cepat")
                            elif 'hovering' in shape_desc:
                                features.append("melayang")
                            
                            feature_text = ", ".join(features[:3]) if features else "ciri bervariasi"
                            
                            ranking_data.append({
                                'Rank': i,
                                'Shape': shape.upper(),
                                'Count': count,
                                'Percentage': f"{percentage:.1f}%",
                                'Characteristics': feature_text
                            })
                        
                        ranking_df = pd.DataFrame(ranking_data)
                        
                        # Display as interactive table
                        st.dataframe(
                            ranking_df.style.background_gradient(subset=['Count'], cmap='Blues'),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Visualizations
                        st.markdown("### 📈 Visualisasi Data")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = px.pie(
                                values=shape_counts.values[:8], 
                                names=shape_counts.index[:8],
                                title=f'Distribusi Bentuk UFO di {location_name}',
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig1.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = px.bar(
                                x=shape_counts.index[:10], 
                                y=shape_counts.values[:10],
                                title='Top 10 Bentuk UFO',
                                labels={'x': 'Shape', 'y': 'Jumlah Penampakan'},
                                color=shape_counts.values[:10],
                                color_continuous_scale='viridis'
                            )
                            fig2.update_layout(showlegend=False)
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Timeline analysis
                        st.markdown("### 📅 Analisis Timeline")
                        yearly_data = filtered_df.groupby(['Dates.Sighted.Year', 'Data.Shape']).size().reset_index(name='count')
                        top_shapes = shape_counts.index[:5]
                        yearly_top = yearly_data[yearly_data['Data.Shape'].isin(top_shapes)]
                        
                        fig3 = px.line(
                            yearly_top,
                            x='Dates.Sighted.Year',
                            y='count',
                            color='Data.Shape',
                            title=f'Tren Penampakan 5 Bentuk Teratas di {location_name}',
                            labels={'Dates.Sighted.Year': 'Tahun', 'count': 'Jumlah Penampakan', 'Data.Shape': 'Bentuk'},
                            markers=True
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                        
                    else:
                        st.error(f"❌ Tidak ditemukan data untuk lokasi: {location_name}")
                        st.info("💡 Coba lokasi lain atau periksa ejaan nama kota/state")
        
        elif not user_query:
            st.warning("⚠️ Silakan masukkan pertanyaan Anda untuk memulai analisis")
    
    elif page == "📍 Analisis Lokasi":
        st.markdown('<p class="sub-header">📍 Analisis Berdasarkan Lokasi</p>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("Pilih Tipe Analisis:", ["City", "State", "Country"])
        
        if analysis_type == "City":
            cities = sorted(df_clean['Location.City'].unique())
            selected = st.selectbox("Pilih Kota:", cities)
            filtered_df = df_clean[df_clean['Location.City'] == selected]
            location_name = selected.title()
        
        elif analysis_type == "State":
            states = sorted(df_clean['Location.State'].unique())
            selected = st.selectbox("Pilih State:", states)
            filtered_df = df_clean[df_clean['Location.State'] == selected]
            location_name = selected
        
        else:  # Country
            countries = sorted(df_clean['Location.Country'].unique())
            selected = st.selectbox("Pilih Negara:", countries)
            filtered_df = df_clean[df_clean['Location.Country'] == selected]
            location_name = selected
        
        if st.button("📊 Analisis", type="primary"):
            if len(filtered_df) > 0:
                shape_counts = filtered_df['Data.Shape'].value_counts()
                total = len(filtered_df)
                
                st.markdown(f"### 📈 Hasil Analisis untuk {location_name}")
                st.info(f"Total penampakan: **{total}** kasus")
                
                # Top shape
                top_shape = shape_counts.index[0]
                top_percentage = (shape_counts.values[0] / total) * 100
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>🏆 Bentuk UFO Paling Sering Muncul</h4>
                    <h2>{top_shape.upper()}</h2>
                    <p>{shape_counts.values[0]} penampakan ({top_percentage:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Rankings
                st.markdown("### 🏅 Peringkat Lengkap")
                
                cols = st.columns(3)
                for i, (shape, count) in enumerate(shape_counts.items()):
                    percentage = (count / total) * 100
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>#{i+1} {shape.upper()}</strong><br>
                            {count} kasus ({percentage:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.bar(x=shape_counts.index[:10], y=shape_counts.values[:10],
                                 title='Top 10 Bentuk UFO',
                                 labels={'x': 'Shape', 'y': 'Count'},
                                 color=shape_counts.values[:10],
                                 color_continuous_scale='plasma')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.pie(values=shape_counts.values[:8], names=shape_counts.index[:8],
                                 title='Distribusi Bentuk (Top 8)',
                                 hole=0.4)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Yearly trend
                st.markdown("### 📅 Tren per Tahun")
                yearly_shape = filtered_df.groupby(['Dates.Sighted.Year', 'Data.Shape']).size().reset_index(name='count')
                
                fig3 = px.line(yearly_shape[yearly_shape['Data.Shape'].isin(shape_counts.index[:5])],
                              x='Dates.Sighted.Year', y='count', color='Data.Shape',
                              title='Tren Penampakan 5 Bentuk Teratas',
                              labels={'Dates.Sighted.Year': 'Year', 'count': 'Sightings'})
                st.plotly_chart(fig3, use_container_width=True)
                
            else:
                st.error("❌ Tidak ada data untuk lokasi yang dipilih")
    
    elif page == "🤖 Model Training":
        st.markdown('<p class="sub-header">🤖 Training Model Neural Network</p>', unsafe_allow_html=True)
        
        st.info("Model akan memprediksi bentuk UFO berdasarkan lokasi dan tahun penampakan")
        
        # Prepare data
        st.markdown("### 📊 Persiapan Data")
        
        # Encode features
        df_model = df_clean.copy()
        
        # Filter out shapes with very few samples (minimum 10 samples per class)
        shape_counts = df_model['Data.Shape'].value_counts()
        valid_shapes = shape_counts[shape_counts >= 10].index
        df_model = df_model[df_model['Data.Shape'].isin(valid_shapes)]
        
        st.info(f"📊 Filtering: Menggunakan {len(valid_shapes)} shapes dengan minimal 10 samples (dari {len(shape_counts)} total shapes)")
        
        if len(df_model) < 100:
            st.error("❌ Dataset terlalu kecil untuk training. Minimal 100 samples diperlukan.")
            st.stop()
        
        # Label encode categorical variables
        le_city = LabelEncoder()
        le_state = LabelEncoder()
        le_country = LabelEncoder()
        le_shape = LabelEncoder()
        
        df_model['city_encoded'] = le_city.fit_transform(df_model['Location.City'])
        df_model['state_encoded'] = le_state.fit_transform(df_model['Location.State'])
        df_model['country_encoded'] = le_country.fit_transform(df_model['Location.Country'])
        df_model['shape_encoded'] = le_shape.fit_transform(df_model['Data.Shape'])
        
        # Features and target
        X = df_model[['city_encoded', 'state_encoded', 'country_encoded', 'Dates.Sighted.Year']].values
        y = df_model['shape_encoded'].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()
        
        # Adjust test_size based on minimum samples
        if min_samples < 5:
            test_size = 0.2
            val_size = 0.5
        else:
            test_size = 0.3
            val_size = 0.5
        
        try:
            # Split data with stratify
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
        except ValueError:
            # If stratify still fails, split without stratify
            st.warning("⚠️ Menggunakan split tanpa stratification karena distribusi kelas tidak merata")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )
        
        st.success(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Show class distribution
        with st.expander("📊 Lihat Distribusi Kelas"):
            class_dist = pd.DataFrame({
                'Shape': le_shape.inverse_transform(unique),
                'Count': counts
            }).sort_values('Count', ascending=False)
            
            fig_dist = px.bar(class_dist, x='Shape', y='Count',
                            title='Distribusi Shape dalam Dataset Training',
                            color='Count',
                            color_continuous_scale='viridis')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Model selection
        st.markdown("### 🎯 Pilih Model untuk Training")
        
        model_choice = st.selectbox("Pilih Model:", 
                                    ["MLP (Multilayer Perceptron)", 
                                     "TabNet (Pretrained-inspired)",
                                     "FT-Transformer (Pretrained)"])
        
        # Recommended hyperparameters for each model
        model_recommendations = {
            "MLP (Multilayer Perceptron)": {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32,
                "description": "MLP bekerja baik dengan learning rate sedang dan batch size standar. Model ini cepat konvergen."
            },
            "TabNet (Pretrained-inspired)": {
                "epochs": 70,
                "learning_rate": 0.0005,
                "batch_size": 64,
                "description": "TabNet memerlukan lebih banyak epochs dan learning rate lebih rendah untuk attention mechanism yang stabil. Batch size lebih besar membantu generalisasi."
            },
            "FT-Transformer (Pretrained)": {
                "epochs": 80,
                "learning_rate": 0.0001,
                "batch_size": 32,
                "description": "Transformer membutuhkan training lebih lama dengan learning rate kecil untuk menghindari instabilitas. Batch size sedang optimal untuk self-attention."
            }
        }
        
        # Show recommendation
        rec = model_recommendations[model_choice]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; border-left: 4px solid #4b2f88;">
            <h4>💡 Rekomendasi Hyperparameter untuk {model_choice}</h4>
            <p><strong>📊 Epochs:</strong> {rec['epochs']}</p>
            <p><strong>📈 Learning Rate:</strong> {rec['learning_rate']}</p>
            <p><strong>📦 Batch Size:</strong> {rec['batch_size']}</p>
            <p><small>ℹ️ {rec['description']}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to use recommended settings
        use_recommended = st.checkbox("✨ Gunakan Setting yang Direkomendasikan", value=True)
        
        if use_recommended:
            epochs = rec['epochs']
            learning_rate = rec['learning_rate']
            batch_size = rec['batch_size']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Epochs", epochs, help="Jumlah iterasi training")
            with col2:
                st.metric("Learning Rate", learning_rate, help="Kecepatan pembelajaran model")
            with col3:
                st.metric("Batch Size", batch_size, help="Jumlah sample per batch")
        else:
            st.markdown("#### ⚙️ Custom Hyperparameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Epochs:", 10, 150, rec['epochs'], 
                                 help="Lebih banyak epochs = training lebih lama tapi potensi akurasi lebih tinggi")
            with col2:
                learning_rate = st.select_slider("Learning Rate:", 
                                                options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                                                value=rec['learning_rate'],
                                                help="Learning rate kecil = training stabil, besar = cepat tapi risiko overshoot")
            with col3:
                batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], 
                                        index=[16, 32, 64, 128].index(rec['batch_size']),
                                        help="Batch size besar = training lebih stabil, kecil = lebih banyak update")
        
        if st.button("🚀 Start Training", type="primary"):
            # Create datasets
            train_dataset = UFODataset(X_train, y_train)
            val_dataset = UFODataset(X_val, y_val)
            test_dataset = UFODataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
            
            # Initialize model
            input_size = X_train.shape[1]
            num_classes = len(np.unique(y))
            
            if model_choice == "MLP (Multilayer Perceptron)":
                model = MLPClassifier(input_size, num_classes)
            elif model_choice == "TabNet (Pretrained-inspired)":
                model = SimpleTabNet(input_size, num_classes)
            else:
                model = FTTransformer(input_size, num_classes)
            
            st.info(f"🔧 Training {model_choice}...")
            
            # Train
            history = train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)
            
            # Save to session state
            if 'models' not in st.session_state:
                st.session_state.models = {}
            
            st.session_state.models[model_choice] = {
                'model': model,
                'history': history,
                'label_encoder': le_shape,
                'scaler': scaler,
                'test_loader': test_loader,
                'y_test': y_test
            }
            
            st.success(f"✅ Training selesai! Model disimpan ke session.")
            
            # Auto-save to disk
            with st.spinner("💾 Menyimpan model ke disk..."):
                metadata = {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'test_size': len(X_test),
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'best_val_acc': max(history['val_acc']),
                    'final_test_acc': None  # Will be calculated in evaluation
                }
                
                save_path = save_model(model, model_choice, le_shape, scaler, history, metadata,
                                       extra_encoders={
                                           'le_city': le_city,
                                           'le_state': le_state,
                                           'le_country': le_country
                                       })
                st.success(f"✅ Model tersimpan di: `{save_path}`")
            
            st.balloons()
            
            # Plot training history
            col1, col2 = st.columns(2)
            
            with col1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss', mode='lines'))
                fig_loss.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'))
                fig_loss.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(y=history['train_acc'], name='Train Acc', mode='lines'))
                fig_acc.add_trace(go.Scatter(y=history['val_acc'], name='Val Acc', mode='lines'))
                fig_acc.update_layout(title='Accuracy Over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy (%)')
                st.plotly_chart(fig_acc, use_container_width=True)
    
    elif page == "📈 Evaluasi Model":
        st.markdown('<p class="sub-header">📈 Evaluasi Performa Model</p>', unsafe_allow_html=True)
        
        if 'models' in st.session_state and len(st.session_state.models) > 0:
            
            # Comparison of all models first
            if len(st.session_state.models) >= 2:
                st.markdown("### 🔬 Perbandingan Semua Model")
                
                comparison_data = []
                all_reports = {}
                
                for model_name, model_data in st.session_state.models.items():
                    model = model_data['model']
                    test_loader = model_data['test_loader']
                    le_shape = model_data['label_encoder']
                    
                    # Get predictions
                    y_pred, y_true = evaluate_model(model, test_loader, le_shape)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    history = model_data['history']
                    best_val_acc = max(history['val_acc'])
                    final_train_acc = history['train_acc'][-1]
                    final_val_acc = history['val_acc'][-1]
                    final_train_loss = history['train_loss'][-1]
                    final_val_loss = history['val_loss'][-1]
                    
                    # Calculate overfitting indicator
                    overfitting = final_train_acc - final_val_acc
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Test Accuracy': accuracy * 100,
                        'Best Val Acc': best_val_acc,
                        'Final Train Acc': final_train_acc,
                        'Final Val Acc': final_val_acc,
                        'Train Loss': final_train_loss,
                        'Val Loss': final_val_loss,
                        'Overfitting': overfitting,
                        'Macro F1': report['macro avg']['f1-score'] * 100,
                        'Weighted F1': report['weighted avg']['f1-score'] * 100
                    })
                    
                    all_reports[model_name] = {'y_pred': y_pred, 'y_true': y_true, 'report': report}
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(
                    comp_df.style.format({
                        'Test Accuracy': '{:.2f}%',
                        'Best Val Acc': '{:.2f}%',
                        'Final Train Acc': '{:.2f}%',
                        'Final Val Acc': '{:.2f}%',
                        'Train Loss': '{:.4f}',
                        'Val Loss': '{:.4f}',
                        'Overfitting': '{:.2f}%',
                        'Macro F1': '{:.2f}%',
                        'Weighted F1': '{:.2f}%'
                    }).background_gradient(subset=['Test Accuracy', 'Macro F1', 'Weighted F1'], cmap='Greens')
                    .background_gradient(subset=['Overfitting'], cmap='Reds'),
                    use_container_width=True
                )
                
                # Best model highlight
                best_model_idx = comp_df['Test Accuracy'].idxmax()
                best_model_name = comp_df.loc[best_model_idx, 'Model']
                best_accuracy = comp_df.loc[best_model_idx, 'Test Accuracy']
                
                st.markdown(f"""
                <div class="info-box">
                    <h3>🏆 Model Terbaik: {best_model_name}</h3>
                    <h2>{best_accuracy:.2f}% Test Accuracy</h2>
                    <p>Model ini memberikan performa terbaik pada test set.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations for comparison
                st.markdown("### 📊 Visualisasi Perbandingan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_acc = px.bar(
                        comp_df, 
                        x='Model', 
                        y=['Test Accuracy', 'Best Val Acc', 'Final Train Acc'],
                        title='Perbandingan Accuracy',
                        barmode='group',
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
                    )
                    fig_acc.update_layout(yaxis_title='Accuracy (%)', legend_title='Metric')
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    fig_f1 = px.bar(
                        comp_df,
                        x='Model',
                        y=['Macro F1', 'Weighted F1'],
                        title='Perbandingan F1-Score',
                        barmode='group',
                        color_discrete_sequence=['#4facfe', '#00f2fe']
                    )
                    fig_f1.update_layout(yaxis_title='F1-Score (%)', legend_title='Metric')
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                # Loss comparison
                fig_loss = px.bar(
                    comp_df,
                    x='Model',
                    y=['Train Loss', 'Val Loss'],
                    title='Perbandingan Loss',
                    barmode='group',
                    color_discrete_sequence=['#fa709a', '#fee140']
                )
                fig_loss.update_layout(yaxis_title='Loss', legend_title='Type')
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Overfitting analysis
                st.markdown("### 🎯 Analisis Overfitting")
                fig_overfit = px.bar(
                    comp_df,
                    x='Model',
                    y='Overfitting',
                    title='Indikator Overfitting (Train Acc - Val Acc)',
                    color='Overfitting',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_overfit.add_hline(y=5, line_dash="dash", line_color="red", 
                                     annotation_text="Threshold (5%)")
                fig_overfit.update_layout(yaxis_title='Overfitting Gap (%)')
                st.plotly_chart(fig_overfit, use_container_width=True)
                
                st.info("💡 **Interpretasi Overfitting:** Gap < 5% = Good, 5-10% = Acceptable, > 10% = Overfitting")
                
                # Side-by-side confusion matrices
                if len(st.session_state.models) <= 3:
                    st.markdown("### 🔲 Perbandingan Confusion Matrix")
                    cols = st.columns(len(st.session_state.models))
                    
                    for idx, (model_name, report_data) in enumerate(all_reports.items()):
                        with cols[idx]:
                            cm = confusion_matrix(report_data['y_true'], report_data['y_pred'])
                            le_shape = st.session_state.models[model_name]['label_encoder']
                            class_names = le_shape.classes_
                            
                            fig_cm = px.imshow(
                                cm,
                                labels=dict(x="Predicted", y="Actual"),
                                x=class_names,
                                y=class_names,
                                title=f'{model_name}',
                                color_continuous_scale='Blues',
                                text_auto=True,
                                aspect='auto'
                            )
                            fig_cm.update_layout(height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)
                
                st.markdown("---")
            
            # Individual model evaluation
            st.markdown("### 🔍 Evaluasi Detail per Model")
            model_names = list(st.session_state.models.keys())
            selected_model = st.selectbox("Pilih Model untuk Analisis Detail:", model_names)
            
            if st.button("📊 Tampilkan Evaluasi Detail", type="primary"):
                model_data = st.session_state.models[selected_model]
                model = model_data['model']
                test_loader = model_data['test_loader']
                y_test = model_data['y_test']
                le_shape = model_data['label_encoder']
                
                st.info(f"🔍 Evaluating {selected_model}...")
                
                # Predictions
                y_pred, y_true = evaluate_model(model, test_loader, le_shape)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                st.markdown(f"""
                <div class="info-box">
                    <h3>📊 Overall Accuracy - {selected_model}</h3>
                    <h2>{accuracy*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Classification Report
                st.markdown("### 📋 Classification Report")
                class_names = le_shape.classes_
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
                
                # Convert to dataframe
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(
                    report_df.style.format("{:.3f}").background_gradient(cmap='YlGn'), 
                    use_container_width=True
                )
                
                # Confusion Matrix
                st.markdown("### 🔲 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                
                fig_cm = px.imshow(cm, 
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=class_names,
                                   y=class_names,
                                   title=f'Confusion Matrix - {selected_model}',
                                   color_continuous_scale='Blues',
                                   text_auto=True)
                fig_cm.update_layout(height=600)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Training History
                st.markdown("### 📈 Training History")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    history = model_data['history']
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss', 
                                                 mode='lines', line=dict(color='blue')))
                    fig_loss.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss', 
                                                 mode='lines', line=dict(color='red')))
                    fig_loss.update_layout(title='Loss Curve', 
                                         xaxis_title='Epoch', 
                                         yaxis_title='Loss',
                                         hovermode='x unified')
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                with col2:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(y=history['train_acc'], name='Train Accuracy', 
                                                mode='lines', line=dict(color='green')))
                    fig_acc.add_trace(go.Scatter(y=history['val_acc'], name='Validation Accuracy', 
                                                mode='lines', line=dict(color='orange')))
                    fig_acc.update_layout(title='Accuracy Curve', 
                                        xaxis_title='Epoch', 
                                        yaxis_title='Accuracy (%)',
                                        hovermode='x unified')
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                # Per-class performance
                st.markdown("### 📊 Per-Class Performance")
                
                metrics_data = []
                for i, class_name in enumerate(class_names):
                    if class_name in report:
                        metrics_data.append({
                            'Shape': class_name,
                            'Precision': report[class_name]['precision'],
                            'Recall': report[class_name]['recall'],
                            'F1-Score': report[class_name]['f1-score'],
                            'Support': int(report[class_name]['support'])
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                fig_metrics = go.Figure()
                fig_metrics.add_trace(go.Bar(name='Precision', x=metrics_df['Shape'], 
                                           y=metrics_df['Precision'], marker_color='lightblue'))
                fig_metrics.add_trace(go.Bar(name='Recall', x=metrics_df['Shape'], 
                                           y=metrics_df['Recall'], marker_color='lightgreen'))
                fig_metrics.add_trace(go.Bar(name='F1-Score', x=metrics_df['Shape'], 
                                           y=metrics_df['F1-Score'], marker_color='lightyellow'))
                
                fig_metrics.update_layout(
                    barmode='group',
                    title='Metrics by Class',
                    xaxis_title='Shape',
                    yaxis_title='Score',
                    height=500
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
                
        else:
            st.warning("⚠️ Belum ada model yang di-training. Silakan train model terlebih dahulu di halaman 'Model Training'.")
    
    elif page == "💾 Model Management":
        st.markdown('<p class="sub-header">💾 Manajemen Model</p>', unsafe_allow_html=True)
        
        st.info("📦 Kelola model yang tersimpan di disk dan session")
        
        # Two columns for saved models and session models
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💿 Model di Disk")
            saved_models = load_saved_models()
            
            if saved_models:
                st.success(f"✅ Ditemukan {len(saved_models)} model tersimpan")
                
                for idx, model_info in enumerate(saved_models):
                    with st.expander(f"📦 {model_info['name']} - {model_info['timestamp']}"):
                        st.write(f"**Path:** `{model_info['path']}`")
                        st.write(f"**Num Classes:** {model_info['num_classes']}")
                        
                        metadata = model_info['metadata']
                        if 'best_val_acc' in metadata:
                            st.write(f"**Best Val Acc:** {metadata['best_val_acc']:.2f}%")
                        if 'epochs' in metadata:
                            st.write(f"**Epochs:** {metadata['epochs']}")
                        if 'learning_rate' in metadata:
                            st.write(f"**Learning Rate:** {metadata['learning_rate']}")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if st.button(f"📥 Load", key=f"load_{idx}", use_container_width=True):
                                with st.spinner("Loading model..."):
                                    # Need to prepare test data first
                                    if os.path.exists(dataset_path):
                                        df = load_data(dataset_path)
                                        df_clean = preprocess_data(df)
                                        
                                        # Prepare data (same as training)
                                        df_model = df_clean.copy()
                                        shape_counts = df_model['Data.Shape'].value_counts()
                                        valid_shapes = shape_counts[shape_counts >= 10].index
                                        df_model = df_model[df_model['Data.Shape'].isin(valid_shapes)]
                                        
                                        # Load model
                                        input_size = 4  # city, state, country, year
                                        model, le_shape, scaler, history, metadata = load_model_from_disk(
                                            model_info['path'], input_size
                                        )
                                        
                                        # Prepare test data
                                        le_city = LabelEncoder()
                                        le_state = LabelEncoder()
                                        le_country = LabelEncoder()
                                        
                                        df_model['city_encoded'] = le_city.fit_transform(df_model['Location.City'])
                                        df_model['state_encoded'] = le_state.fit_transform(df_model['Location.State'])
                                        df_model['country_encoded'] = le_country.fit_transform(df_model['Location.Country'])
                                        df_model['shape_encoded'] = le_shape.transform(df_model['Data.Shape'])
                                        
                                        X = df_model[['city_encoded', 'state_encoded', 'country_encoded', 'Dates.Sighted.Year']].values
                                        y = df_model['shape_encoded'].values
                                        X = scaler.transform(X)
                                        
                                        # Split for test
                                        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                        test_dataset = UFODataset(X_test, y_test)
                                        test_loader = DataLoader(test_dataset, batch_size=32, drop_last=False)
                                        
                                        # Add to session state
                                        if 'models' not in st.session_state:
                                            st.session_state.models = {}
                                        
                                        st.session_state.models[model_info['name']] = {
                                            'model': model,
                                            'history': history,
                                            'label_encoder': le_shape,
                                            'scaler': scaler,
                                            'test_loader': test_loader,
                                            'y_test': y_test
                                        }
                                        
                                        st.success(f"✅ Model '{model_info['name']}' loaded to session!")
                                        st.rerun()
                                    else:
                                        st.error("Dataset not found!")
                        
                        with col_b:
                            if st.button(f"🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
                                import shutil
                                shutil.rmtree(model_info['path'])
                                st.success("Model deleted!")
                                st.rerun()
            else:
                st.info("📭 Belum ada model yang tersimpan")
                st.caption("Train model terlebih dahulu di menu 'Model Training'")
        
        with col2:
            st.markdown("### 💻 Model di Session")
            
            if 'models' in st.session_state and len(st.session_state.models) > 0:
                st.success(f"✅ {len(st.session_state.models)} model di session")
                
                for model_name in st.session_state.models.keys():
                    with st.expander(f"🔷 {model_name}"):
                        model_data = st.session_state.models[model_name]
                        history = model_data['history']
                        
                        st.write(f"**Best Val Acc:** {max(history['val_acc']):.2f}%")
                        st.write(f"**Final Train Acc:** {history['train_acc'][-1]:.2f}%")
                        st.write(f"**Epochs Trained:** {len(history['train_acc'])}")
                        
                        if st.button(f"🗑️ Remove from Session", key=f"remove_{model_name}", use_container_width=True):
                            del st.session_state.models[model_name]
                            st.success(f"Model '{model_name}' removed!")
                            st.rerun()
            else:
                st.info("📭 Belum ada model di session")
                st.caption("Model akan muncul di sini setelah training atau load dari disk")
        
        # Additional tools
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            if st.button("🔄 Refresh List", use_container_width=True):
                st.rerun()
        
        with col_t2:
            if st.button("🗑️ Clear All Session Models", use_container_width=True, type="primary"):
                if 'models' in st.session_state:
                    del st.session_state.models
                    st.success("All session models cleared!")
                    st.rerun()
        
        # Export/Import functionality
        st.markdown("---")
        st.markdown("### 📤 Export/Import")
        
        st.info("💡 **Tip:** Folder `saved_models/` berisi semua model yang tersimpan. Anda bisa backup folder ini untuk menyimpan model secara permanen.")
        
        if os.path.exists(MODELS_DIR):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(MODELS_DIR)
                for filename in filenames
            ) / (1024 * 1024)  # Convert to MB
            
            st.metric("Total Storage Used", f"{total_size:.2f} MB")

else:
    st.markdown("""
    <div class="info-box">
        <h3>❌ Dataset Tidak Ditemukan!</h3>
        <p>Aplikasi ini memerlukan file dataset untuk berfungsi.</p>
        
        <h4>📁 Instruksi Setup:</h4>
        <ol>
            <li>Pastikan file <code>ufo_sightings.csv</code> berada di folder yang sama dengan <code>app.py</code></li>
            <li>Struktur folder yang benar:
                <pre>
project/
├── app.py
└── ufo_sightings.csv
                </pre>
            </li>
            <li>Refresh halaman setelah file dataset tersedia</li>
        </ol>
        
        <h4>📊 Format Data yang Diperlukan:</h4>
        <p>Dataset harus memiliki kolom-kolom berikut:</p>
        <ul>
            <li><code>Location.City</code> - Nama kota</li>
            <li><code>Location.State</code> - Kode state</li>
            <li><code>Location.Country</code> - Negara</li>
            <li><code>Data.Shape</code> - Bentuk UFO</li>
            <li><code>Data.Description excerpt</code> - Deskripsi penampakan</li>
            <li><code>Dates.Sighted.Year</code> - Tahun penampakan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(f"⚠️ File `{dataset_path}` tidak ditemukan di direktori saat ini.")
    st.info(f"📍 Direktori saat ini: `{os.getcwd()}`")