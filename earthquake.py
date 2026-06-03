import streamlit as st

# 반드시 최상단
st.set_page_config(
    page_title="지진 위험도 분석",
    layout="wide"
)

import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# -------------------------------
# 데이터 불러오기
# -------------------------------
df_new = joblib.load("Earthquake.csv")

# 테스트용 (나중에 제거 가능)
df_new = df_new.head(100)

# 컬럼명 통일
df_new = df_new.rename(columns={
    "significance": "영향도",
    "magnitudo": "규모",
    "depth": "진원깊이",
    "latitude": "위도",
    "longitude": "경도"
})

# 좌표 없는 데이터 제거
df_new = df_new.dropna(subset=["위도", "경도"])

# -------------------------------
# 모델 불러오기
# -------------------------------
model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# 군집 예측
# -------------------------------
X = df_new[["영향도", "규모", "진원깊이"]]
X_scaled = scaler.transform(X)

df_new["cluster"] = model.predict(X_scaled)

# -------------------------------
# 제목
# -------------------------------
st.title("🌍 지진 위험도 분석 시스템")

# -------------------------------
# 위험도 매핑
# -------------------------------
risk_dict = {
    0: "매우 낮음",
    1: "낮음",
    2: "중간",
    3: "높음"
}

# -------------------------------
# 세션 상태 초기화
# -------------------------------
if "analyze" not in st.session_state:
    st.session_state.analyze = False

# -------------------------------
# 사용자 입력
# -------------------------------
lat = st.number_input(
    "위도 입력",
    value=35.0,
    format="%.4f"
)

lon = st.number_input(
    "경도 입력",
    value=129.0,
    format="%.4f"
)

# -------------------------------
# 분석 버튼
# -------------------------------
if st.button("위험도 분석"):
    st.session_state.analyze = True
    st.session_state.lat = lat
    st.session_state.lon = lon

# -------------------------------
# 분석 실행
# -------------------------------
if st.session_state.analyze:

    lat = st.session_state.lat
    lon = st.session_state.lon

    near_df = df_new[
        (df_new["위도"] >= lat - 5) &
        (df_new["위도"] <= lat + 5) &
        (df_new["경도"] >= lon - 5) &
        (df_new["경도"] <= lon + 5)
    ]

    if len(near_df) == 0:
        st.warning("주변 지진 데이터가 없습니다.")

    else:

        cluster_ratio = near_df["cluster"].value_counts(normalize=True)

        main_cluster = int(cluster_ratio.idxmax())

        risk = risk_dict.get(
            main_cluster,
            f"군집 {main_cluster}"
        )

        st.success(f"예상 위험도 : {risk}")

        # -------------------------------
        # 지도 생성
        # -------------------------------
        m = folium.Map(
            location=[lat, lon],
            zoom_start=5
        )

        colors = {
            0: "red",
            1: "blue",
            2: "green",
            3: "orange"
        }

        for _, row in df_new.iterrows():

            cluster = int(row["cluster"])

            color = colors.get(
                cluster,
                "gray"
            )

            folium.CircleMarker(
                location=[
                    row["위도"],
                    row["경도"]
                ],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)

        # 입력 위치 표시
        folium.Marker(
            location=[lat, lon],
            popup="입력 위치",
            icon=folium.Icon(color="black")
        ).add_to(m)

        st.subheader("지진 분포 지도")

        st_folium(
            m,
            width=1000,
            height=600,
            returned_objects=[]
        )
