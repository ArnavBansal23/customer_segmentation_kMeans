import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
    }
    .main { background-color: #0e0e16; }
    .block-container { padding: 2rem 3rem; }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a45;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .segment-card {
        background: #1a1a2e;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .segment-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #fff;
    }
    .segment-tip {
        font-size: 0.82rem;
        color: #aaa;
        margin-top: 4px;
    }
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        border-bottom: 1px solid #2a2a45;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stSlider > div > div { background: #4ECDC4 !important; }
    .predict-result {
        background: #1a1a2e;
        border: 1px solid #4ECDC4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#A8E6CF', '#C3A6FF']
SEGMENT_COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#A8E6CF', '#C3A6FF']

SEGMENT_INFO = {
    0: ("High Income, High Spenders",  "VIP customers — reward with loyalty programs & exclusive deals.",       "#FF6B6B"),
    1: ("High Income, Low Spenders",   "Hard to convert — use premium product campaigns.",                      "#4ECDC4"),
    2: ("Young, High Spenders",        "Trend-driven — target via social media & flash sales.",                 "#FFE66D"),
    3: ("Low Income, Low Spenders",    "Budget segment — focus on discounts & value offers.",                   "#A8E6CF"),
    4: ("Average Customers",           "Largest group — nurture with personalized recommendations.",            "#C3A6FF"),
}

# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.rename(columns={
            'Annual Income (k$)': 'Annual_Income_k',
            'Spending Score (1-100)': 'Spending_Score'
        }, inplace=True)
    else:
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'CustomerID': range(1, n+1),
            'Gender': np.random.choice(['Male','Female'], n, p=[0.44,0.56]),
            'Age': np.concatenate([np.random.randint(18,35,60), np.random.randint(30,50,80), np.random.randint(45,70,60)]),
            'Annual_Income_k': np.concatenate([np.random.randint(15,40,50), np.random.randint(40,75,80), np.random.randint(70,140,70)]),
            'Spending_Score': np.concatenate([np.random.randint(60,100,40), np.random.randint(40,80,50),
                                              np.random.randint(1,40,50),   np.random.randint(15,50,30), np.random.randint(60,100,30)])
        })
    return df

@st.cache_data
def run_model(df_hash, k):
    df = st.session_state['df'].copy()
    features = ['Age', 'Annual_Income_k', 'Spending_Score']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df['Cluster'])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    return df, kmeans, scaler, pca, sil, X_scaled

# ── Header ────────────────────────────────────────────────────
st.markdown("## 🛍️ Customer Segmentation")
st.markdown("<p style='color:#888; margin-top:-0.8rem;'>K-Means Clustering — ML Lab Project</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=['csv'])
    k_value = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app performs **customer segmentation** using K-Means clustering on the Mall Customers dataset.

    **Features used:**
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)

    **Steps:**
    1. Load & explore data
    2. Scale features
    3. Find optimal K
    4. Train K-Means
    5. Visualize & interpret
    """)

# ── Load data ─────────────────────────────────────────────────
df_raw = load_data(uploaded_file)
st.session_state['df'] = df_raw
df, kmeans, scaler, pca, sil_score, X_scaled = run_model(id(df_raw), k_value)

# ── Metric cards ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df)}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{k_value}</div><div class="metric-label">Clusters (K)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{sil_score:.3f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{round(kmeans.inertia_)}</div><div class="metric-label">Inertia</div></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Elbow & Silhouette", "🎯 Clusters", "🔮 Predict"])

# ════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.describe().round(2), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Gender Distribution</div>', unsafe_allow_html=True)
        gender_counts = df_raw['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index,
                                           autopct='%1.1f%%', colors=['#4ECDC4','#FF6B6B'],
                                           textprops={'color':'white', 'fontsize':12})
        ax.set_title('Gender Split', color='white', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#1a1a2e')
    cols = ['Age', 'Annual_Income_k', 'Spending_Score']
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']
    for ax, col, color in zip(axes, cols, colors):
        ax.set_facecolor('#1a1a2e')
        ax.hist(df_raw[col], bins=20, color=color, edgecolor='#0e0e16', linewidth=0.5)
        ax.set_title(col.replace('_',' '), color='white', fontweight='bold')
        ax.set_xlabel('Value', color='#888')
        ax.set_ylabel('Frequency', color='#888')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2a45')
        ax.grid(alpha=0.15, color='#444')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Scatter Plots</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a2e')
    for ax in axes: ax.set_facecolor('#1a1a2e')
    axes[0].scatter(df_raw['Annual_Income_k'], df_raw['Spending_Score'],
                    alpha=0.7, color='#4ECDC4', edgecolors='#0e0e16', linewidth=0.3, s=60)
    axes[0].set_title('Annual Income vs Spending Score', color='white', fontweight='bold')
    axes[0].set_xlabel('Annual Income (k$)', color='#888')
    axes[0].set_ylabel('Spending Score', color='#888')
    axes[0].tick_params(colors='#888')
    for spine in axes[0].spines.values(): spine.set_edgecolor('#2a2a45')
    axes[0].grid(alpha=0.15, color='#444')

    axes[1].scatter(df_raw['Age'], df_raw['Spending_Score'],
                    alpha=0.7, color='#FF6B6B', edgecolors='#0e0e16', linewidth=0.3, s=60)
    axes[1].set_title('Age vs Spending Score', color='white', fontweight='bold')
    axes[1].set_xlabel('Age', color='#888')
    axes[1].set_ylabel('Spending Score', color='#888')
    axes[1].tick_params(colors='#888')
    for spine in axes[1].spines.values(): spine.set_edgecolor('#2a2a45')
    axes[1].grid(alpha=0.15, color='#444')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 2 — Elbow & Silhouette
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Finding Optimal K</div>', unsafe_allow_html=True)

    inertia_vals, sil_vals = [], []
    K_range = range(2, 11)
    features = ['Age', 'Annual_Income_k', 'Spending_Score']
    X_s = StandardScaler().fit_transform(df_raw[features])
    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_s)
        inertia_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(X_s, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a2e')
    for ax in axes: ax.set_facecolor('#1a1a2e')

    axes[0].plot(list(K_range), inertia_vals, 'o-', color='#FF6B6B', linewidth=2.5, markersize=8)
    axes[0].axvline(x=k_value, color='#4ECDC4', linestyle='--', linewidth=1.8, label=f'K={k_value}')
    axes[0].set_title('Elbow Method', color='white', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)', color='#888')
    axes[0].set_ylabel('Inertia', color='#888')
    axes[0].tick_params(colors='#888')
    for spine in axes[0].spines.values(): spine.set_edgecolor('#2a2a45')
    axes[0].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[0].grid(alpha=0.15, color='#444')

    bar_colors = [('#4ECDC4' if k == k_value else '#2a2a45') for k in K_range]
    axes[1].bar(list(K_range), sil_vals, color=bar_colors, edgecolor='#0e0e16', linewidth=0.5)
    axes[1].set_title('Silhouette Scores', color='white', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)', color='#888')
    axes[1].set_ylabel('Silhouette Score', color='#888')
    axes[1].tick_params(colors='#888')
    for spine in axes[1].spines.values(): spine.set_edgecolor('#2a2a45')
    axes[1].grid(alpha=0.15, color='#444', axis='y')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    results_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia': [round(i, 1) for i in inertia_vals],
        'Silhouette Score': [round(s, 4) for s in sil_vals]
    })
    results_df['Selected'] = results_df['K'].apply(lambda x: '← selected' if x == k_value else '')
    st.dataframe(results_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Clusters
# ════════════════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Cluster Visualizations</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#1a1a2e')
        for ax in axes: ax.set_facecolor('#1a1a2e')
        patches = [mpatches.Patch(color=COLORS[i % len(COLORS)], label=f'Cluster {i}') for i in range(k_value)]

        for i in range(k_value):
            mask = df['Cluster'] == i
            axes[0].scatter(df.loc[mask,'PCA1'], df.loc[mask,'PCA2'],
                            c=COLORS[i % len(COLORS)], s=65, alpha=0.85,
                            edgecolors='#0e0e16', linewidth=0.3)
        centers_pca = pca.transform(kmeans.cluster_centers_)
        axes[0].scatter(centers_pca[:,0], centers_pca[:,1],
                        c='white', s=220, marker='*', zorder=5)
        axes[0].set_title('PCA — 2D Cluster View', color='white', fontweight='bold')
        axes[0].set_xlabel('PC1', color='#888')
        axes[0].set_ylabel('PC2', color='#888')
        axes[0].tick_params(colors='#888')
        for spine in axes[0].spines.values(): spine.set_edgecolor('#2a2a45')
        axes[0].legend(handles=patches, facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        axes[0].grid(alpha=0.15, color='#444')

        for i in range(k_value):
            mask = df['Cluster'] == i
            axes[1].scatter(df.loc[mask,'Annual_Income_k'], df.loc[mask,'Spending_Score'],
                            c=COLORS[i % len(COLORS)], s=65, alpha=0.85,
                            edgecolors='#0e0e16', linewidth=0.3)
        axes[1].set_title('Income vs Spending Score', color='white', fontweight='bold')
        axes[1].set_xlabel('Annual Income (k$)', color='#888')
        axes[1].set_ylabel('Spending Score', color='#888')
        axes[1].tick_params(colors='#888')
        for spine in axes[1].spines.values(): spine.set_edgecolor('#2a2a45')
        axes[1].legend(handles=patches, facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        axes[1].grid(alpha=0.15, color='#444')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">Cluster Profiles</div>', unsafe_allow_html=True)
        profile = df.groupby('Cluster')[['Age','Annual_Income_k','Spending_Score']].mean().round(1)
        profile['Count'] = df['Cluster'].value_counts().sort_index()
        st.dataframe(profile, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Segment Insights</div>', unsafe_allow_html=True)
        for i in range(min(k_value, len(SEGMENT_INFO))):
            name, tip, color = SEGMENT_INFO[i]
            st.markdown(f"""
            <div class="segment-card" style="border-left-color:{color}">
                <div class="segment-title" style="color:{color}">Cluster {i} — {name}</div>
                <div class="segment-tip">{tip}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Cluster Sizes</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        counts = df['Cluster'].value_counts().sort_index()
        bars = ax.bar([f'C{i}' for i in counts.index],
                      counts.values, color=[COLORS[i % len(COLORS)] for i in counts.index],
                      edgecolor='#0e0e16', linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color='white')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values(): spine.set_edgecolor('#2a2a45')
        ax.grid(alpha=0.15, color='#444', axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 4 — Predict
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Predict Customer Segment</div>', unsafe_allow_html=True)
    st.markdown("Enter a new customer's details below to find out which segment they belong to.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_input = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        income_input = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=60)
    with col3:
        spending_input = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    if st.button("🔮 Predict Segment", use_container_width=True):
        new_customer = np.array([[age_input, income_input, spending_input]])
        new_scaled = scaler.transform(new_customer)
        predicted_cluster = kmeans.predict(new_scaled)[0]

        name, tip, color = SEGMENT_INFO.get(predicted_cluster,
            (f"Cluster {predicted_cluster}", "Custom cluster based on your K value.", COLORS[predicted_cluster % len(COLORS)]))

        st.markdown(f"""
        <div class="predict-result">
            <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">Predicted Segment</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:{color};">
                Cluster {predicted_cluster} — {name}
            </div>
            <div style="color:#aaa; margin-top:0.5rem; font-size:0.9rem;">{tip}</div>
            <div style="margin-top:1rem; display:flex; justify-content:center; gap:2rem;">
                <div><span style="color:#888; font-size:0.8rem;">AGE</span><br>
                     <span style="color:white; font-weight:600;">{age_input}</span></div>
                <div><span style="color:#888; font-size:0.8rem;">INCOME</span><br>
                     <span style="color:white; font-weight:600;">${income_input}k</span></div>
                <div><span style="color:#888; font-size:0.8rem;">SPENDING</span><br>
                     <span style="color:white; font-weight:600;">{spending_input}/100</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#555; font-size:0.8rem;'>ML Lab Project — Customer Segmentation using K-Means Clustering</p>", unsafe_allow_html=True)
