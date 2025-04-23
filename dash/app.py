
import streamlit as st
st.set_page_config(page_title="상원이의 집을 찾아서", layout="wide")
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("../data/our_df.csv")  # 실제 our_df로 저장한 파일명
    return df

our_df = load_data()

st.sidebar.title("🏠 페이지 이동")
page = st.sidebar.radio("Go to", ["1. 소개 & EDA", "2. 지도 시각화", "3. 조건 필터링", "4. Lasso 회귀 분석", "5. 최종 추천"])

isu_lat, isu_lon = 42.0267, -93.6465

if page == "1. 소개 & EDA":
    st.title("📖 상원이의 집을 찾아서")
    st.markdown("4명의 아이오와 유학생이 조건에 맞는 집을 찾습니다.")

    st.markdown("""
    - 🎓 학교 근처여야 하고  
    - 💰 월세는 저렴해야 하고  
    - 🛏️ 방/욕실 수는 넉넉해야 하고  
    - 🔥 난방/에어컨, 전기 시스템도 중요하고  
    - 🚗 차고, 🏞️ 공원, 🧱 울타리, 도로 포장 상태까지 봅니다.
    """)

    st.subheader("📊 주요 변수 분포")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(our_df, x="BedroomAbvGr", title="방 개수 분포"), use_container_width=True)
        st.plotly_chart(px.histogram(our_df, x="GarageCars", title="차고 수 분포"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(our_df, x="TotalFullBath", title="Full Bath 수"), use_container_width=True)
        st.plotly_chart(px.histogram(our_df, x="dist_to_ISU", title="학교 거리 분포"), use_container_width=True)

elif page == "2. 지도 시각화":
    st.title("🗺️ 지도 기반 집 위치 보기")
    m = folium.Map(location=[isu_lat, isu_lon], zoom_start=14, tiles='CartoDB positron')
    folium.Circle(location=[isu_lat, isu_lon], radius=2000, color='blue',
                  fill=True, fill_opacity=0.05, popup='ISU 2km').add_to(m)
    folium.Marker([isu_lat, isu_lon], tooltip='Iowa State University',
                  icon=folium.Icon(color='blue', icon='university', prefix='fa')).add_to(m)

    cluster = MarkerCluster().add_to(m)
    for _, row in our_df.iterrows():
        if row['dist_to_ISU'] <= 2000:
            popup_html = f"""
            <b>{row['HouseStyle']}층</b><br>
            🛏 {row['BedroomAbvGr']} Bed / 🛁 {row['TotalFullBath']} Bath<br>
            🚗 Garage: {row['GarageCars']} / {row['GarageArea']} sqft<br>
            📏 거리: {int(row['dist_to_ISU'])}m
            """
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                popup=popup_html,
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(cluster)

    st_data = st_folium(m, width=1000)

elif page == "3. 조건 필터링":
    st.title("🔍 조건으로 집 필터링")
    max_dist = st.slider("📏 학교에서 거리 (m)", 500, 5000, 2000)
    min_bed = st.slider("🛏 최소 방 개수", 1, 5, 2)
    min_bath = st.slider("🛁 최소 욕실 개수", 1.0, 4.0, 2.0)
    garage = st.checkbox("🚗 차고 있어야 함")

    filtered = our_df[
        (our_df['dist_to_ISU'] <= max_dist) &
        (our_df['BedroomAbvGr'] >= min_bed) &
        (our_df['TotalFullBath'] >= min_bath)
    ]
    if garage:
        filtered = filtered[our_df['GarageCars'] > 0]

    st.subheader(f"🏠 조건을 만족하는 집: {len(filtered)}개")
    st.dataframe(filtered[['BedroomAbvGr', 'TotalFullBath', 'GarageCars', 'dist_to_ISU']])

elif page == "4. Lasso 회귀 분석":
    st.title("📈 Lasso 회귀 분석")
    st.markdown("좋은 집 조건이 실제 집 품질에 어떤 영향을 미치는지 Lasso 회귀로 분석합니다.")

    X = our_df[['BedroomAbvGr', 'TotalFullBath', 'GarageCars', 'GarageArea', '2ndFlrSF', '1stFlrSF', 'dist_to_ISU']]
    y = our_df['OverallCond']

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=0))
    ])
    pipeline.fit(X, y)
    coef = pipeline.named_steps['lasso'].coef_
    feature_importance = pd.Series(coef, index=X.columns).sort_values()
    st.bar_chart(feature_importance)

elif page == "5. 최종 추천":
    st.title("🎉 최종 추천 집!")
    final = our_df[
        (our_df['dist_to_ISU'] <= 2000) &
        (our_df['BedroomAbvGr'] >= 3) &
        (our_df['TotalFullBath'] >= 2) &
        (our_df['GarageCars'] > 0)
    ].sort_values(by=['OverallCond', 'GarageArea'], ascending=False).head(1)

    st.subheader("🏠 이 집이 가장 적합해요!")
    st.dataframe(final)
