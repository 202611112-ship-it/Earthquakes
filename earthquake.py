import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

with open("Earthquake.csv", "rb") as f:
    data = f.read(100)

st.write(data)
st.stop()
# -------------------------------
# 데이터 불러오기
# -------------------------------
df_new = pd.read_csv("Earthquake.csv", encoding="euc-kr")

# 컬럼명 통일
df_new = df_new.rename(columns={
    "significance": "영향도",
    "magnitudo": "규모",
    "depth": "진원깊이",
    "latitude": "위도",
    "longitude": "경도"
})

# -------------------------------
# 모델 불러오기
# -------------------------------
model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("scaler.pkl")

# 군집 예측
X = df_new[["영향도", "규모", "진원깊이"]]
X_scaled = scaler.transform(X)

df_new["cluster"] = model.predict(X_scaled)

# -------------------------------
# 화면 설정
# -------------------------------
st.set_page_config(
    page_title="지진 위험도 분석",
    layout="wide"
)

st.title("🌍 지진 위험도 분석 시스템")

# 위험도 매핑
risk_dict = {
    0: "매우 낮음",
    1: "낮음",
    2: "중간",
    3: "높음"
}

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

        # ---------------------------
        # 지도 생성
        # ---------------------------
        m = folium.Map(
            location=[lat, lon],
            zoom_start=5
        )

        df_sample = df_new.sample(
            min(5000, len(df_new)),
            random_state=42
        )

        colors = {
            0: "red",
            1: "blue",
            2: "green",
            3: "orange"
        }

        for _, row in df_sample.iterrows():

            cluster = int(row["cluster"])

            color = colors.get(cluster, "gray")

            folium.CircleMarker(
                location=[row["위도"], row["경도"]],
                radius=3,
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

        # 입력 위치 표시
        folium.Marker(
            location=[lat, lon],
            popup="입력 위치",
            icon=folium.Icon(color="black")
        ).add_to(m)

        st_folium(
            m,
            width=1000,
            height=600
        )
