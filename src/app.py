import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Causal Marketing Analyzer',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Marketing Causal Impact Analyzer')
st.markdown('Upload your marketing data to find which channel **truly caused** your sales.')

uploaded_file = st.file_uploader(
    'Upload your marketing CSV file',
    type=['csv'],
    help='CSV must have columns: TV, Radio, Newspaper, Sales'
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success('File uploaded successfully! ✅')

    st.subheader('Raw Data Preview')
    st.dataframe(df.head(10))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Rows',        len(df))
    col2.metric('Total TV Spend',    f"{df['TV'].sum():.0f}")
    col3.metric('Total Radio Spend', f"{df['Radio'].sum():.0f}")
    col4.metric('Total Sales',       f"{df['Sales'].sum():.0f}")

    st.subheader('📈 Correlation Analysis')
    fig, ax = plt.subplots(figsize=(8, 4))
    corr = df.corr()['Sales'].drop('Sales').sort_values()
    colors = ['coral' if x < 0.3 else 'steelblue' for x in corr]
    corr.plot(kind='barh', color=colors, ax=ax)
    ax.set_title('Correlation with Sales')
    ax.set_xlabel('Correlation Coefficient')
    st.pyplot(fig)
    plt.close()

    st.subheader('🔍 Causal Impact Analysis')
    campaign_start = st.slider(
        'Campaign started at row number:',
        min_value=20,
        max_value=len(df) - 20,
        value=len(df) // 2
    )

    df['week'] = pd.date_range(
        start='2021-01-01',
        periods=len(df),
        freq='W'
    )
    df = df.set_index('week')

    pre_data  = df.iloc[:campaign_start]
    post_data = df.iloc[campaign_start:]

    X_pre  = sm.add_constant(pre_data[['TV', 'Radio', 'Newspaper']])
    X_post = sm.add_constant(post_data[['TV', 'Radio', 'Newspaper']])

    model         = sm.OLS(pre_data['Sales'], X_pre).fit()
    predicted     = model.predict(X_post)
    point_effect  = post_data['Sales'].values - predicted.values
    cum_effect    = point_effect.cumsum()

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    axes[0].plot(df.index, df['Sales'],
                 label='Actual Sales', color='steelblue')
    axes[0].plot(post_data.index, predicted,
                 label='Predicted (no campaign)',
                 color='coral', linestyle='--')
    axes[0].axvline(x=post_data.index[0],
                    color='red', linestyle='--',
                    label='Campaign Start')
    axes[0].set_title('Actual vs Predicted Sales')
    axes[0].legend()

    axes[1].plot(post_data.index, point_effect, color='green')
    axes[1].axhline(y=0, color='black')
    axes[1].axvline(x=post_data.index[0],
                    color='red', linestyle='--')
    axes[1].set_title('Point Effect per Week')

    axes[2].plot(post_data.index, cum_effect, color='purple')
    axes[2].axhline(y=0, color='black')
    axes[2].set_title('Cumulative Effect')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader('📊 Campaign Summary')
    avg_actual    = post_data['Sales'].mean()
    avg_predicted = predicted.mean()
    avg_effect    = point_effect.mean()

    col1, col2, col3 = st.columns(3)
    col1.metric('Actual Sales/Week',    f'{avg_actual:.2f}')
    col2.metric('Without Campaign',     f'{avg_predicted:.2f}')
    col3.metric('Campaign Effect/Week', f'{avg_effect:.2f}',
                delta=f'{(avg_effect/avg_predicted*100):.1f}%')

    st.subheader('💰 ROI Per Channel')
    X      = df.reset_index()[['TV', 'Radio', 'Newspaper']]
    y      = df.reset_index()['Sales']
    lasso  = LassoCV(cv=5).fit(X, y)
    coefs  = dict(zip(X.columns, lasso.coef_))
    total_coef   = sum(abs(v) for v in coefs.values())
    total_sales  = y.sum()
    spend  = {
        'TV':        X['TV'].sum(),
        'Radio':     X['Radio'].sum(),
        'Newspaper': X['Newspaper'].sum()
    }

    roi_data = []
    for ch, coef in coefs.items():
        pct        = abs(coef) / total_coef if total_coef > 0 else 0
        attributed = total_sales * pct
        roi        = attributed / spend[ch] if spend[ch] > 0 else 0
        roi_data.append({
            'Channel':        ch,
            'ROI':            round(roi, 3),
            'Attribution %':  round(pct * 100, 1)
        })

    roi_df = pd.DataFrame(roi_data).sort_values('ROI', ascending=False)
    st.dataframe(roi_df, use_container_width=True)

else:
    st.info('Please upload a CSV file to begin analysis.')
    st.markdown('**Expected columns:** TV, Radio, Newspaper, Sales')