import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Propensity Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * {
    color: #ccd6f6 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #112240 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 24px rgba(100,255,218,0.05);
}
div[data-testid="metric-container"] label {
    color: #8892b0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #64ffda !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #ccd6f6 !important;
}

/* Section headers */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #ccd6f6 !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    color: #8892b0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #64ffda !important;
    border-bottom: 2px solid #64ffda !important;
}

/* DataFrames */
.stDataFrame {
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

/* Expanders */
details {
    background: #112240 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
summary {
    color: #64ffda !important;
    font-weight: 600 !important;
}

p, li, span, div {
    color: #ccd6f6;
}

.insight-box {
    background: linear-gradient(135deg, #112240 0%, #0d2137 100%);
    border-left: 3px solid #64ffda;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #a8b2d8 !important;
    line-height: 1.6;
}

.section-title {
    font-family: 'Playfair Display', serif;
    color: #ccd6f6;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}

.caption-text {
    font-size: 12px;
    color: #8892b0;
    font-style: italic;
    margin-top: 4px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #112240 0%, #0a3d62 50%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    color: #ccd6f6 !important;
    margin: 0;
}
.hero-banner p {
    color: #8892b0;
    font-size: 15px;
    margin: 8px 0 0 0;
}
.hero-accent {
    color: #64ffda !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #64ffda 0%, #0070f3 100%);
    color: #0a0f1e;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(100,255,218,0.3);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #233554 0%, #112240 100%);
    color: #64ffda;
    font-weight: 600;
    border: 1px solid #64ffda;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
}

/* Alert boxes */
.stAlert {
    background: #112240 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#64ffda',
    'secondary': '#0070f3',
    'accent':    '#f7b731',
    'danger':    '#ff6b6b',
    'bg':        '#0a0f1e',
    'card':      '#112240',
    'text':      '#ccd6f6',
    'muted':     '#8892b0',
    'palette':   ['#64ffda','#0070f3','#f7b731','#ff6b6b','#a855f7','#34d399','#fb923c'],
}

plt.rcParams.update({
    'figure.facecolor':  COLORS['bg'],
    'axes.facecolor':    COLORS['card'],
    'axes.edgecolor':    '#1e3a5f',
    'axes.labelcolor':   COLORS['text'],
    'xtick.color':       COLORS['muted'],
    'ytick.color':       COLORS['muted'],
    'text.color':        COLORS['text'],
    'grid.color':        '#1e3a5f',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
    'axes.titlecolor':   COLORS['text'],
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
})

# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('UniversalBank.csv')
    df['Experience'] = df['Experience'].clip(lower=0)
    return df

def get_features(df):
    drop_cols = ['ID', 'ZIP Code', 'Personal Loan']
    existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing)
    y = df['Personal Loan'] if 'Personal Loan' in df.columns else None
    return X, y

@st.cache_resource
def train_models(df):
    X, y = get_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Oversample minority class to handle class imbalance
    train_df = X_train.copy()
    train_df['__target__'] = y_train.values
    maj = train_df[train_df['__target__'] == 0]
    mn  = train_df[train_df['__target__'] == 1]
    mn_up = resample(mn, replace=True, n_samples=len(maj), random_state=42)
    balanced = pd.concat([maj, mn_up])
    X_train_res = balanced.drop('__target__', axis=1)
    y_train_res = balanced['__target__']

    models = {
        'Decision Tree':         DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':         RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosted Tree': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    }
    results = {}
    for name, clf in models.items():
        clf.fit(X_train_res, y_train_res)
        y_pred_train = clf.predict(X_train)
        y_pred_test  = clf.predict(X_test)
        y_prob       = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_prob)
        roc_auc      = auc(fpr, tpr)
        results[name] = {
            'clf':        clf,
            'y_test':     y_test,
            'y_pred':     y_pred_test,
            'y_prob':     y_prob,
            'fpr':        fpr,
            'tpr':        tpr,
            'auc':        roc_auc,
            'train_acc':  accuracy_score(y_train, y_pred_train),
            'test_acc':   accuracy_score(y_test,  y_pred_test),
            'precision':  precision_score(y_test,  y_pred_test),
            'recall':     recall_score(y_test,     y_pred_test),
            'f1':         f1_score(y_test,         y_pred_test),
            'cm':         confusion_matrix(y_test,  y_pred_test),
            'feature_names': list(X.columns),
        }
    return results, X_test, y_test, X_train, y_train

def style_fig(fig):
    fig.patch.set_facecolor(COLORS['bg'])
    return fig

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
results, X_test, y_test, X_train, y_train = train_models(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Loan Propensity Intelligence**")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["📊 Overview & EDA",
         "🔍 Deep Dive Analysis",
         "🤖 ML Models",
         "🎯 Predict New Customers"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.markdown(f"**Total Customers:** {len(df):,}")
    st.markdown(f"**Loan Takers:** {df['Personal Loan'].sum():,} ({df['Personal Loan'].mean()*100:.1f}%)")
    st.markdown(f"**Features:** {len(df.columns)-1}")
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#8892b0;'>Built for Universal Bank Marketing · 2024</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":

    st.markdown("""
    <div class="hero-banner">
        <h1>🏦 Loan Propensity <span class="hero-accent">Intelligence Dashboard</span></h1>
        <p>Universal Bank · Personal Loan Marketing Analytics · Head of Marketing View</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Customers", f"{len(df):,}")
    with c2: st.metric("Loan Acceptances", f"{df['Personal Loan'].sum():,}", f"{df['Personal Loan'].mean()*100:.1f}% rate")
    with c3: st.metric("Avg. Income", f"${df['Income'].mean():.0f}K")
    with c4: st.metric("Avg. CC Spend", f"${df['CCAvg'].mean():.2f}K/mo")
    with c5: st.metric("CD Account Holders", f"{df['CD Account'].sum():,}", f"{df['CD Account'].mean()*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Target distribution + Income distribution
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="section-title">Target Variable Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sizes  = [df['Personal Loan'].value_counts()[0], df['Personal Loan'].value_counts()[1]]
        labels = [f'No Loan\n{sizes[0]:,}\n({sizes[0]/len(df)*100:.1f}%)',
                  f'Accepted Loan\n{sizes[1]:,}\n({sizes[1]/len(df)*100:.1f}%)']
        wedges, texts = ax.pie(
            sizes, labels=labels,
            colors=[COLORS['muted'], COLORS['primary']],
            startangle=90, wedgeprops=dict(width=0.65, edgecolor='#0a0f1e', linewidth=2),
            textprops={'color': COLORS['text'], 'fontsize': 10}
        )
        ax.set_title('Personal Loan Acceptance', pad=16)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Only 9.6% of customers accepted the loan in the last campaign — a classic class imbalance. This means your marketing ROI is highly dependent on targeting the right 10%. Oversampling (random minority resampling) is applied during model training to handle this imbalance.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Income Distribution by Loan Status</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        loan0 = df[df['Personal Loan']==0]['Income']
        loan1 = df[df['Personal Loan']==1]['Income']
        ax.hist(loan0, bins=35, color=COLORS['muted'],   alpha=0.75, label='No Loan',       edgecolor='#0a0f1e', linewidth=0.5)
        ax.hist(loan1, bins=35, color=COLORS['primary'], alpha=0.85, label='Accepted Loan', edgecolor='#0a0f1e', linewidth=0.5)
        ax.axvline(loan0.mean(), color=COLORS['muted'],   linestyle='--', linewidth=1.5, label=f'No Loan Mean: ${loan0.mean():.0f}K')
        ax.axvline(loan1.mean(), color=COLORS['primary'], linestyle='--', linewidth=1.5, label=f'Loan Mean: ${loan1.mean():.0f}K')
        ax.set_xlabel('Annual Income ($000s)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Income Distribution: Loan Acceptors vs Non-Acceptors')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Loan acceptors earn an average of $144K vs $60K for non-acceptors — a 2.4× difference. Income is the single strongest predictor. Focus your campaign on customers earning above $100K for maximum conversion.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Education breakdown + Family size
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Loan Acceptance by Education Level</div>', unsafe_allow_html=True)
        edu_map  = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Prof'}
        edu_data = df.copy()
        edu_data['Education Label'] = edu_data['Education'].map(edu_map)
        grp = edu_data.groupby('Education Label')['Personal Loan'].agg(['sum','count'])
        grp['rate'] = grp['sum'] / grp['count'] * 100
        grp = grp.loc[['Undergrad','Graduate','Advanced/Prof']]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(grp.index, grp['rate'],
                      color=[COLORS['secondary'], COLORS['accent'], COLORS['primary']],
                      edgecolor='#0a0f1e', linewidth=1, width=0.55)
        for bar, val, cnt in zip(bars, grp['rate'], grp['sum']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}%\n(n={cnt})', ha='center', va='bottom', fontsize=9, color=COLORS['text'])
        ax.set_ylabel('Loan Acceptance Rate (%)')
        ax.set_title('Acceptance Rate by Education Level')
        ax.set_ylim(0, grp['rate'].max() * 1.3)
        ax.grid(axis='y', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Advanced/Professional degree holders show the highest loan acceptance rate (~13%). Graduate and undergraduate customers follow. Tailor your messaging: professionals respond to prestige and financial growth narratives.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Loan Acceptance by Family Size</div>', unsafe_allow_html=True)
        fam_grp = df.groupby('Family')['Personal Loan'].agg(['sum','count'])
        fam_grp['rate'] = fam_grp['sum'] / fam_grp['count'] * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_fam = [COLORS['palette'][i] for i in range(len(fam_grp))]
        bars = ax.bar(fam_grp.index.astype(str), fam_grp['rate'],
                      color=colors_fam, edgecolor='#0a0f1e', linewidth=1, width=0.55)
        for bar, val, cnt in zip(bars, fam_grp['rate'], fam_grp['sum']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f'{val:.1f}%\n(n={cnt})', ha='center', va='bottom', fontsize=9, color=COLORS['text'])
        ax.set_xlabel('Family Size (members)')
        ax.set_ylabel('Loan Acceptance Rate (%)')
        ax.set_title('Acceptance Rate by Family Size')
        ax.set_ylim(0, fam_grp['rate'].max() * 1.3)
        ax.grid(axis='y', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Families of 3–4 members have significantly higher loan acceptance rates, likely driven by home/education financing needs. Consider family-centric messaging for mid-size households in your next campaign.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Correlation heatmap + CCAvg
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        corr_cols = ['Age','Income','CCAvg','Family','Mortgage','Education',
                     'Securities Account','CD Account','Online','CreditCard','Personal Loan']
        corr = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
                    linewidths=0.5, linecolor='#0a0f1e', ax=ax,
                    annot_kws={'size':8, 'color':'white'},
                    cbar_kws={'shrink':0.7})
        ax.set_title('Correlation Matrix — All Features', pad=14)
        ax.tick_params(axis='x', rotation=45)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Income (0.50) and CCAvg (0.37) are the top correlated features with Personal Loan. CD Account (0.32) is a strong behavioural signal. Age and Experience are near-zero — demographic age alone does not predict loan uptake.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">CC Spend vs Loan Acceptance</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5.5))
        loan0 = df[df['Personal Loan']==0]['CCAvg']
        loan1 = df[df['Personal Loan']==1]['CCAvg']
        parts = ax.violinplot([loan0, loan1], positions=[0,1],
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([COLORS['muted'], COLORS['primary']][i])
            pc.set_alpha(0.75)
        for part in ['cmeans','cmedians','cbars','cmins','cmaxes']:
            parts[part].set_color(COLORS['text'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Loan', 'Accepted Loan'])
        ax.set_ylabel('Monthly CC Spend ($000s)')
        ax.set_title('Credit Card Spend Distribution')
        ax.grid(axis='y', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Customers who accepted loans have a median CC spend nearly 3× higher. High-spend credit card users signal financial activity — a top segment for targeted loan offers.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Deep Dive Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Deep Dive Analysis":

    st.markdown('<h2 style="font-family:Playfair Display,serif;color:#ccd6f6;">🔍 Deep Dive Segment Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892b0;margin-bottom:24px;">Uncover hidden customer segments to craft hyper-personalised campaigns.</p>', unsafe_allow_html=True)

    # ── CD Account + Securities Account
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">CD & Securities Accounts vs Loan</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        for i, (col, title) in enumerate([('CD Account','CD Account'), ('Securities Account','Securities Account')]):
            grp = df.groupby(col)['Personal Loan'].mean() * 100
            axes[i].bar(['No','Yes'], grp.values,
                        color=[COLORS['muted'], COLORS['primary']],
                        edgecolor='#0a0f1e', width=0.5)
            for j, v in enumerate(grp.values):
                axes[i].text(j, v+0.3, f'{v:.1f}%', ha='center', fontsize=10, color=COLORS['text'], fontweight='bold')
            axes[i].set_title(title, fontsize=11)
            axes[i].set_ylabel('Acceptance Rate (%)' if i==0 else '')
            axes[i].set_ylim(0, grp.max()*1.4)
            axes[i].grid(axis='y', alpha=0.3)
        fig.suptitle('Acceptance Rate by Account Type', fontsize=12, fontweight='bold', y=1.01)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">CD account holders convert at ~20% — more than double the average. They already trust the bank with savings, making them the warmest leads. Securities account holders show a more modest lift.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Online Banking & CreditCard Usage</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        for i, (col, title) in enumerate([('Online','Online Banking'), ('CreditCard','Bank Credit Card')]):
            grp = df.groupby(col)['Personal Loan'].mean() * 100
            axes[i].bar(['No','Yes'], grp.values,
                        color=[COLORS['muted'], COLORS['accent']],
                        edgecolor='#0a0f1e', width=0.5)
            for j, v in enumerate(grp.values):
                axes[i].text(j, v+0.3, f'{v:.1f}%', ha='center', fontsize=10, color=COLORS['text'], fontweight='bold')
            axes[i].set_title(title, fontsize=11)
            axes[i].set_ylabel('Acceptance Rate (%)' if i==0 else '')
            axes[i].set_ylim(0, grp.max()*1.4)
            axes[i].grid(axis='y', alpha=0.3)
        fig.suptitle('Digital Engagement vs Loan Acceptance', fontsize=12, fontweight='bold', y=1.01)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Online banking users show a marginally higher acceptance rate. Bank credit card users are not significantly different — suggesting digital channel preference alone is not a strong signal without income context.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Income × Education heatmap
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="section-title">Loan Rate: Income Bracket × Education</div>', unsafe_allow_html=True)
        df2 = df.copy()
        df2['Income Bracket'] = pd.cut(df2['Income'], bins=[0,50,100,150,225],
                                       labels=['<$50K','$50–100K','$100–150K','$150K+'])
        df2['Education Label'] = df2['Education'].map({1:'Undergrad',2:'Graduate',3:'Advanced/Prof'})
        pivot = df2.groupby(['Income Bracket','Education Label'])['Personal Loan'].mean() * 100
        pivot = pivot.unstack('Education Label')
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, linecolor='#0a0f1e', ax=ax,
                    cbar_kws={'label':'Acceptance Rate (%)', 'shrink':0.8},
                    annot_kws={'size':10, 'weight':'bold'})
        ax.set_title('Loan Acceptance Rate (%) by Income & Education', pad=14)
        ax.set_xlabel('')
        ax.set_ylabel('Income Bracket')
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">The highest acceptance rates (40–60%+) sit in the top-right: high-income advanced-degree holders. This is your "golden segment" — a small pool but with extremely high conversion probability. Personalise with premium loan products.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Mortgage Holders vs Non-Holders</div>', unsafe_allow_html=True)
        df3 = df.copy()
        df3['Has Mortgage'] = (df3['Mortgage'] > 0).map({True:'Has Mortgage', False:'No Mortgage'})
        grp = df3.groupby('Has Mortgage')['Personal Loan'].agg(['sum','count'])
        grp['rate'] = grp['sum'] / grp['count'] * 100
        fig, ax = plt.subplots(figsize=(5, 4.5))
        bars = ax.barh(grp.index, grp['rate'],
                       color=[COLORS['secondary'], COLORS['danger']],
                       edgecolor='#0a0f1e', height=0.45)
        for bar, val, cnt in zip(bars, grp['rate'], grp['sum']):
            ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%  (n={cnt})', va='center', fontsize=10, color=COLORS['text'])
        ax.set_xlabel('Acceptance Rate (%)')
        ax.set_title('Loan Rate: Mortgage Status')
        ax.set_xlim(0, grp['rate'].max() * 1.5)
        ax.grid(axis='x', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">Mortgage holders convert at a higher rate — they are already comfortable with large financial commitments. Use this as a behavioural filter when layering targeting criteria for your campaign.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Age distribution + Prescriptive summary
    st.markdown('<div class="section-title">Age Profile of Loan Acceptors</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.6, 1])

    with col1:
        df4 = df.copy()
        df4['Age Group'] = pd.cut(df4['Age'], bins=[20,30,40,50,60,70],
                                  labels=['20s','30s','40s','50s','60s'])
        age_grp = df4.groupby('Age Group')['Personal Loan'].agg(['sum','count'])
        age_grp['rate'] = age_grp['sum'] / age_grp['count'] * 100
        fig, ax1 = plt.subplots(figsize=(8, 4.2))
        ax2 = ax1.twinx()
        bars = ax1.bar(age_grp.index.astype(str), age_grp['count'],
                       color=COLORS['muted'], alpha=0.5, label='Total Customers', width=0.5)
        line, = ax2.plot(age_grp.index.astype(str), age_grp['rate'],
                         color=COLORS['primary'], linewidth=2.5, marker='o', markersize=8, label='Acceptance Rate %')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Number of Customers', color=COLORS['muted'])
        ax2.set_ylabel('Acceptance Rate (%)', color=COLORS['primary'])
        ax1.set_title('Customer Volume & Loan Acceptance Rate by Age Group')
        lines = [mpatches.Patch(color=COLORS['muted'], alpha=0.5, label='Customer Count'), line]
        ax1.legend(handles=lines, loc='upper left', fontsize=9)
        ax1.grid(axis='y', alpha=0.2)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#112240,#0d2137);border:1px solid #1e3a5f;border-radius:12px;padding:20px;">
            <div style="color:#64ffda;font-size:14px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">📋 Prescriptive Action Plan</div>
            <div style="font-size:13px;color:#a8b2d8;line-height:1.8;">
                <b style="color:#ccd6f6;">Tier 1 Target (Highest ROI)</b><br>
                • Income $100K+ <br>
                • CD Account holder<br>
                • Advanced/Prof education<br>
                • Family size 3–4<br><br>
                <b style="color:#ccd6f6;">Tier 2 Target (Good ROI)</b><br>
                • Income $60–100K<br>
                • CC Spend $3K+/mo<br>
                • Graduate educated<br>
                • Has mortgage<br><br>
                <b style="color:#ccd6f6;">Channel Strategy</b><br>
                • Digital-first for online users<br>
                • Relationship manager for Tier 1<br>
                • Personalise by family need
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="insight-box">Age alone shows minimal impact — loan acceptance is fairly flat across decades. This reinforces the importance of wealth and behavioural signals (income, CC spend, CD account) over pure demographic profiling in your targeting model.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML Models
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":

    st.markdown('<h2 style="font-family:Playfair Display,serif;color:#ccd6f6;">🤖 Classification Model Performance</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892b0;margin-bottom:24px;">Decision Tree · Random Forest · Gradient Boosted Tree — trained with minority oversampling on 80% of data.</p>', unsafe_allow_html=True)

    # ── Metrics Table
    st.markdown('<div class="section-title">📊 Model Performance Summary</div>', unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rows.append({
            'Model':           name,
            'Train Acc':       f"{r['train_acc']*100:.2f}%",
            'Test Acc':        f"{r['test_acc']*100:.2f}%",
            'Precision':       f"{r['precision']*100:.2f}%",
            'Recall':          f"{r['recall']*100:.2f}%",
            'F1 Score':        f"{r['f1']*100:.2f}%",
            'ROC-AUC':         f"{r['auc']:.4f}",
        })
    metrics_df = pd.DataFrame(rows)

    def highlight_best(col):
        vals = col.str.rstrip('%').astype(float)
        best = vals.max()
        return ['background-color: #1a3a1a; color: #64ffda; font-weight:700;' if v == best else '' for v in vals]

    numeric_cols = ['Train Acc','Test Acc','Precision','Recall','F1 Score','ROC-AUC']
    styled = metrics_df.style.apply(highlight_best, subset=numeric_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown('<div class="insight-box">Highlighted cells show the best value per metric. Gradient Boosted Tree typically delivers the highest AUC and F1 score — ideal for ranking customers by loan propensity. Random Forest balances precision and recall well for batch campaigns.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curve (single chart, all models)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">ROC Curves — All Models</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        roc_colors = [COLORS['primary'], COLORS['accent'], COLORS['danger']]
        for (name, r), color in zip(results.items(), roc_colors):
            ax.plot(r['fpr'], r['tpr'], color=color, linewidth=2.5,
                    label=f"{name}  (AUC = {r['auc']:.3f})")
        ax.plot([0,1],[0,1], color=COLORS['muted'], linestyle='--', linewidth=1.2, label='Random Classifier')
        ax.fill_between([0,1],[0,1], alpha=0.05, color=COLORS['muted'])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve Comparison — All Classification Models', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.2)
        ax.grid(alpha=0.3)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1.02])
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="insight-box">The ROC-AUC score measures how well each model separates loan takers from non-takers across all thresholds. A score near 1.0 means near-perfect ranking — critical for prioritising your outreach list when budgets are limited.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Feature Importance (Best Model)</div>', unsafe_allow_html=True)
        # Pick best model by AUC
        best_name = max(results, key=lambda k: results[k]['auc'])
        best_clf  = results[best_name]['clf']
        feat_names = results[best_name]['feature_names']
        importances = best_clf.feature_importances_
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        colors_fi = [COLORS['primary'] if v == fi_df['Importance'].max() else COLORS['secondary'] for v in fi_df['Importance']]
        ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors_fi, edgecolor='#0a0f1e', height=0.65)
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Feature Importance\n{best_name}', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        style_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(f'<div class="insight-box">For {best_name}, Income and CCAvg dominate. These are the levers to pull in campaign targeting. Features with near-zero importance (e.g. Online, CreditCard) can be deprioritised as filters.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices
    st.markdown('<div class="section-title">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    model_names = list(results.keys())
    cm_colors = [COLORS['primary'], COLORS['accent'], COLORS['danger']]

    for col, name, color in zip(cols, model_names, cm_colors):
        with col:
            r  = results[name]
            cm = r['cm']
            total = cm.sum()
            labels_pct = np.array([[f"{v}\n({v/total*100:.1f}%)" for v in row] for row in cm])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(cm, annot=labels_pct, fmt='', cmap='Blues',
                        linewidths=1, linecolor='#0a0f1e', ax=ax,
                        xticklabels=['Pred: No','Pred: Yes'],
                        yticklabels=['True: No','True: Yes'],
                        cbar=False,
                        annot_kws={'size':11, 'weight':'bold', 'color':'#0a0f1e'})
            ax.set_title(name, fontsize=11, color=color)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            style_fig(fig)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f'<div class="insight-box" style="font-size:11px;"><b>TP:</b> {tp} ({tp/total*100:.1f}%) · <b>FP:</b> {fp} ({fp/total*100:.1f}%)<br><b>FN:</b> {fn} ({fn/total*100:.1f}%) · <b>TN:</b> {tn} ({tn/total*100:.1f}%)</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Predict New Customers
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict New Customers":

    st.markdown('<h2 style="font-family:Playfair Display,serif;color:#ccd6f6;">🎯 Predict Loan Propensity for New Customers</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892b0;margin-bottom:24px;">Upload a CSV file (without the Personal Loan column) to score your customer list and download results.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### 📁 Upload Customer Data")
        uploaded = st.file_uploader(
            "Upload CSV file (same structure as training data, without 'Personal Loan' column)",
            type=['csv']
        )

        st.markdown("#### ⚙️ Choose Model")
        model_choice = st.selectbox(
            "Select Classification Model",
            list(results.keys()),
            index=2
        )
        threshold = st.slider(
            "Prediction Probability Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Lower threshold = cast wider net. Higher = more confident leads."
        )

    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#112240,#0d2137);border:1px solid #1e3a5f;border-radius:12px;padding:20px;margin-top:32px;">
            <div style="color:#64ffda;font-size:13px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">📌 File Format Requirements</div>
            <div style="font-size:12px;color:#a8b2d8;line-height:1.9;">
                Required columns (in any order):<br>
                <code style="background:#0a0f1e;padding:2px 6px;border-radius:4px;color:#64ffda;">ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard</code><br><br>
                ✅ Do NOT include <code style="background:#0a0f1e;padding:2px 6px;border-radius:4px;color:#f7b731;">Personal Loan</code> column<br>
                ✅ Use same scale as training data<br>
                ✅ Download sample test file below
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Download sample test file
    try:
        with open('test_data_sample.csv', 'rb') as f:
            st.download_button(
                "⬇️ Download Sample Test File",
                data=f.read(),
                file_name="test_data_sample.csv",
                mime="text/csv"
            )
    except:
        pass

    st.markdown("---")

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} customer records.")

            # Preprocess
            new_df['Experience'] = new_df['Experience'].clip(lower=0)
            drop_cols = ['ID','ZIP Code','Personal Loan']
            X_new = new_df.drop(columns=[c for c in drop_cols if c in new_df.columns])

            # Predict
            clf = results[model_choice]['clf']
            proba = clf.predict_proba(X_new)[:,1]
            pred  = (proba >= threshold).astype(int)

            result_df = new_df.copy()
            result_df['Loan_Probability']  = np.round(proba, 4)
            result_df['Personal_Loan_Pred'] = pred
            result_df['Propensity_Segment'] = pd.cut(
                proba, bins=[0, 0.25, 0.50, 0.75, 1.01],
                labels=['Low', 'Medium', 'High', 'Very High']
            )

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total Customers Scored", f"{len(result_df):,}")
            with c2: st.metric("Predicted Loan Takers", f"{pred.sum():,}", f"{pred.mean()*100:.1f}% of file")
            with c3: st.metric("Avg. Loan Probability", f"{proba.mean()*100:.1f}%")
            with c4: st.metric("High/Very High Propensity", f"{(proba>=0.5).sum():,}")

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1.4])

            with col1:
                st.markdown('<div class="section-title">Propensity Segment Breakdown</div>', unsafe_allow_html=True)
                seg_counts = result_df['Propensity_Segment'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(5, 4))
                seg_colors = [COLORS['danger'], COLORS['accent'], COLORS['secondary'], COLORS['primary']]
                bars = ax.bar(seg_counts.index.astype(str), seg_counts.values,
                              color=seg_colors, edgecolor='#0a0f1e', width=0.55)
                for bar, v in zip(bars, seg_counts.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                            f'{v}\n({v/len(result_df)*100:.1f}%)',
                            ha='center', va='bottom', fontsize=9, color=COLORS['text'])
                ax.set_xlabel('Propensity Segment')
                ax.set_ylabel('Number of Customers')
                ax.set_title('Customer Distribution by Loan Propensity')
                ax.grid(axis='y', alpha=0.3)
                style_fig(fig)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col2:
                st.markdown('<div class="section-title">Preview: Top Propensity Customers</div>', unsafe_allow_html=True)
                top_leads = result_df.sort_values('Loan_Probability', ascending=False).head(15)
                display_cols = ['ID','Age','Income','Education','Family','CCAvg',
                                'Loan_Probability','Propensity_Segment','Personal_Loan_Pred']
                display_cols = [c for c in display_cols if c in top_leads.columns]
                st.dataframe(top_leads[display_cols].reset_index(drop=True),
                             use_container_width=True, height=340)

            st.markdown("<br>", unsafe_allow_html=True)

            # Download results
            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Scored Results (CSV)",
                data=csv_out,
                file_name="universal_bank_loan_predictions.csv",
                mime="text/csv"
            )

            st.markdown('<div class="insight-box">Results include Loan_Probability (0–1 score), Personal_Loan_Pred (0/1 at your chosen threshold), and Propensity_Segment (Low/Medium/High/Very High) for easy campaign list segmentation. Sort by Loan_Probability descending to prioritise your outreach.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the required columns and matches the training data format.")
    else:
        st.markdown("""
        <div style="background:#112240;border:1px dashed #1e3a5f;border-radius:12px;padding:40px;text-align:center;color:#8892b0;">
            <div style="font-size:48px;margin-bottom:12px;">📂</div>
            <div style="font-size:16px;font-weight:600;color:#ccd6f6;margin-bottom:8px;">Upload a Customer CSV File to Begin</div>
            <div style="font-size:13px;">Download the sample test file above to see the expected format</div>
        </div>
        """, unsafe_allow_html=True)
