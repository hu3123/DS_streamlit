import io

import streamlit as st

import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import matplotlib as mpl
import seaborn as sns

import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'NanumGothic.ttf'  # 다운받은 폰트 파일 경로
fontprop = fm.FontProperties(fname=font_path)


plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

#plt.rcParams['font.family'] = 'Malgun Gothic'
#plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용 시 마이너스 폰트 깨지는 문제 방지


# 데이터 불러오기
CRE = pd.read_csv('2019_CRE.csv', encoding='cp949')
Anti = pd.read_csv('Antibiotic.csv', encoding='cp949')
Hospital = pd.read_csv('hospitals.csv', encoding='utf-8')
merged_data = pd.read_csv('merged_data.csv', encoding='cp949')

# Streamlit 앱 타이틀
st.title('카바페넴 내성 장내세균(CRE) 감염자 수 예측')

# 사이드바 메뉴
mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화','회귀 시각화', 'CRE 감염자 수 예측'])

if mnu == '설명':
    st.subheader('카바페넴 내성 장내세균의 임상적 의의')
    st.write('''
    2018년부터 카바페넴 내성 장내세균(CRE)가 항생제 내성균 중 가장 큰 문제가 되고 있다. 카바페넴은 다제내성 그람음성균 치료의 마지막 대안이라고 여겨졌던 항생제이다. 이를 통해 세균 질환의 치료 걸림돌이 하나 더 생긴 셈이고 이에 대한 예방이 이루어지고 있다.
    질병관리본부 보고에 따르면 CP-CRE는 2010년 해외 유입으로 처음 발생했으나, 이미 2008년에도 국내 혈액검체에서 CRE가 검출되었다는 보고가 있었다. 
    현재 의료계는 사람 또는 병원내 의료기기를 통한 전파를 주 CRE 감염 경로로 보고 있어, 현재 전파를 막는데만 초점을 두고 있다. 하지만 세균의 돌연변이 속도 및 CRE를 구성하는 세균들이 속한 속을 보았을 때 충분히 국내에서 돌연변이에 의해 발생 가능성이 있다. 하지만 의료계에서는 카바페넴 내성 세균의 발생에 대해서는 관심이 없는 상태이다.
    ''')
    
    st.image('CRE.jpg', caption='카바페넴 내성 장내세균')
    st.markdown('#### 데이터 구조')
    st.markdown('**지표연도** - 기록 년도(2017~2021)')
    st.markdown('**시도** - 전국, 서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 경기, 강원, 충북, 충남, 전북, 전남, 경북, 경남, 제주')
    st.markdown('**시군구** - 전국, 서울, 종로구, 중구, 용산구 .... 제주, 제주시, 서귀포시, 미추홀구')
    st.markdown('**CRE수** - 각 지역의 CRE감염자 수')
    st.markdown('**분모** - 항생제 처방률의 분모로 전체 병원 방문자 수')
    st.markdown('**분자** - 항생제 처방률의 분자로 전체 항생제 처방자 수')
    st.markdown('**항생제 처방률** - 분자/분모')
    st.markdown('**의료기관 반영** - 항생제 처방자 수에 각 지역별 의료기관 비율을 곱한 것')
    hospital = ["상급종합병원", "종합병원", "병원", "요양병원", "의원", "치과병원", "치과의원", "조산원", "한방병원", "한의원", "보건의료원", "보건소", "보건지소", "보건진료소", "약국"]
    for i in range(len(hospital)):
        st.markdown(f"&emsp;{i+1}: {hospital[i]}")

elif mnu == 'EDA':

    st.subheader('EDA')

    st.markdown('- 지역별 의료기관')
    st.dataframe(Hospital.head(10))
    buffer = io.StringIO()
    Hospital.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown('- 지역별 CRE 환자 수')
    st.write(CRE.head(10))
    buffer = io.StringIO()
    CRE.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.markdown('- 지역별 항생제 처방')
    st.write(Anti.head(10))
    buffer = io.StringIO()
    Anti.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown('- 학습 시키기 위해 조건 중복되는 것끼리 merge 진행')
    
    
    st.code('''
Anti = pd.read_csv('Antibiotic.csv', encoding='cp949')
Hospital = pd.read_csv('hospitals.csv', encoding='utf-8')
Hospital = Hospital.rename(columns = {"시도":"병원", "계" : "전국",  "서울특별시" : "서울",  "부산광역시" : "부산",  "대구광역시" : "대구",  "인천광역시" : "인천",  "광주광역시" : "광주",  "대전광역시" : "대전",  "울산광역시" : "울산",  "경기도" : "경기",  "강원특별자치도" : "강원",  "충청북도" : "충북",  "충청남도" : "충남",  "전라북도" : "전북",  "전라남도" : "전남",  "경상북도" : "경북",  "경상남도" : "경남",  "제주특별자치도" : "제주",  "세종특별자치시" : "세종"})

Hos = Hospital.set_index('병원')
Hos_2017 = Hos[Hos['년도'] == 2017].fillna(0)
Hos_2018 = Hos[Hos['년도'] == 2018].fillna(0)
Hos_2019 = Hos[Hos['년도'] == 2019].fillna(0)
Hos_2020 = Hos[Hos['년도'] == 2020].fillna(0)
Hos_2021 = Hos[Hos['년도'] == 2021].fillna(0)

l = ['전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북',
    '충남', '전북', '전남', '경북', '경남', '제주', '세종']

H17 = Hos_2017.reset_index().set_index(['년도', '병원'])
string_columns = [col for col in H17.columns if H17[col].dtype == 'object']
for col in string_columns:
    H17[col] = H17[col].str.replace(',', '').astype('Int32')
H17 = H17/H17.loc[2017, '소계']

H18 = Hos_2018.reset_index().set_index(['년도', '병원'])
string_columns = [col for col in H18.columns if H18[col].dtype == 'object']
for col in string_columns:
    H18[col] = H18[col].str.replace(',', '').astype('Int32')
H18 = H18/H18.loc[2018, '소계']

H19 = Hos_2019.reset_index().set_index(['년도', '병원'])
string_columns = [col for col in H18.columns if H19[col].dtype == 'object']
for col in string_columns:
    H19[col] = H19[col].str.replace(',', '').astype('Int32')
H19 = H19/H19.loc[2019, '소계']

H20 = Hos_2020.reset_index().set_index(['년도', '병원'])
string_columns = [col for col in H20.columns if H20[col].dtype == 'object']
for col in string_columns:
    H20[col] = H20[col].str.replace(',', '').astype('Int32')
H20 = H20/H20.loc[2020, '소계']

H21 = Hos_2021.reset_index().set_index(['년도', '병원'])
string_columns = [col for col in H21.columns if H21[col].dtype == 'object']
for col in string_columns:
    H21[col] = H21[col].str.replace(',', '').astype('Int32')
H21 = H21/H21.loc[2021, '소계']
Hos = pd.concat([H17, H18, H19, H20, H21])
Anti[Anti['지표연도'] >= 2017].reset_index()

CRE_s = []
for i in range(2017, 2025):
    CRE_s.append(pd.read_csv(f'{str(i)}_CRE.csv',  encoding = 'cp949'))
    CRE = CRE_s[0]
for i in range(1,8):
    CRE = pd.concat([CRE, CRE_s[i]])
Anti_data = Anti[Anti['지표연도'] >= 2017]
CRE_data = CRE[CRE['지표연도']<= 2021]

long_name = ['전국', '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시',
    '울산광역시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도',
    '경상북도', '경상남도', '제주특별자치도', '세종특별자치시']
short_name = ['전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북',
    '충남', '전북', '전남', '경북', '경남', '제주', '세종']

for i in range(len(long_name)):
    Anti_data.loc[Anti_data['시도'] == long_name[i], '시도'] = short_name[i]
Anti_data = Anti_data.reset_index()
for i in range(len(Anti_data)):
    if Anti_data.loc[i,'시군구'] == '전체':
        Anti_data.loc[i,'시군구'] = Anti_data.loc[i, '시도']
merged_data = pd.merge(Anti_data, CRE_data, on = ['지표연도', '시도', '시군구'])
merged_data['CRE수'] = pd.to_numeric(merged_data['CRE수'], errors='coerce')
merged_data.dropna()
merged_data = pd.merge(Anti_data, CRE_data, on = ['지표연도', '시도', '시군구'])

N=1
hos_list = ["상급종합병원", "종합병원", "병원", "요양병원", "의원", "치과병원", "치과의원", "조산원", "한방병원", "한의원", "보건의료원", "보건소", "보건지소", "보건진료소", "약국"]
for i in range(len(hos_list)):
    print(i)
    for j in range(len(merged_data)):
        merged_data.loc[j, hos_list[i]] = merged_data.loc[j, '분자'] * (Hos.loc[merged_data.loc[j, '지표연도'], merged_data.loc[j, '시도']][hos_list[i]] ** N)
    
merged_data.to_csv('merged_data.csv', index=False, encoding='cp949')
    ''')

    st.markdown('- 합쳐진 데이터')
    st.write(merged_data.head(10))
    buffer = io.StringIO()
    merged_data.info(buf=buffer)
    st.text(buffer.getvalue())


elif mnu == '시각화':

    # Streamlit 애플리케이션 타이틀
    st.title('병원 데이터 시각화')

    # 주요 칼럼 확인
    st.write(merged_data.columns)

    # 연도별 CRE수의 변화 시각화
    st.header('연도별 CRE수의 변화')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=merged_data, x='지표연도', y='CRE수', ax=ax)
    ax.set_title('연도별 CRE수의 변화')
    ax.set_xlabel('연도')
    ax.set_ylabel('CRE수')
    st.pyplot(fig)

    st.write("조금 시간이 걸립니다.")

    st.header('지역별 CRE 수 (2017년 ~ 2021년 합산)')

    data = merged_data[merged_data['시도'] != '전국']

    # 지역별 CRE 수 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=data, x='시도', y='CRE수', palette='viridis', ax=ax)
    ax.set_title('지역별 CRE 수')
    ax.set_xlabel('지역')
    ax.set_ylabel('CRE 수')
    ax.tick_params(axis='x', labelrotation=45)
    
    st.pyplot(fig)

    # 데이터프레임으로 지역별 CRE 수 출력
    st.write(data.groupby('시도')['CRE수'].sum())

    # 병원 종류별 데이터 시각화 및 지역별 CRE수 출력
    st.header('의료기관 종류별 데이터')

    hos_list = ["상급종합병원", "종합병원", "병원", "요양병원", "의원", "치과병원", "치과의원", "조산원", "한방병원", "한의원", "보건의료원", "보건소", "보건지소", "보건진료소", "약국"]


    for hos in hos_list:
        st.subheader(f'{hos} 수 분포 및 지역별 CRE수')

        # 각 병원 종류별로 그래프를 나란히 출력
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1]})

        # 왼쪽 그래프: 병원 수 / 분자
        sns.barplot(data=data, x='시도', y=data[hos] / data['분자'], palette='viridis', ax=axes[0])
        axes[0].set_title(f'{hos} 수 / 전체 의료기관 수')
        axes[0].set_xlabel('지역')
        axes[0].set_ylabel(f'{hos} 수 / 전체 의료기관 수')
        axes[0].tick_params(axis='x', labelrotation=45)

        # 오른쪽 그래프: 지역별 CRE수
        sns.barplot(data=data, x='시도', y=data[hos], palette='viridis', ax=axes[1])
        axes[1].set_title(f'{hos} 수 / 전체 의료기관 수 * 병원 방문환자 수')
        axes[1].set_xlabel('지역')
        axes[1].set_ylabel(f'{hos} 수 / 전체 의료기관 수 * 병원 방문환자 수')
        axes[1].tick_params(axis='x', labelrotation=45)

        plt.tight_layout()
        st.pyplot(fig)  

elif mnu == '회귀 시각화':

    import statsmodels.api as sm
    from mpl_toolkits.mplot3d import Axes3D

    hos_list = ["상급종합병원", "종합병원", "병원", "요양병원", "의원", "치과병원", "치과의원", "조산원", "한방병원", "한의원", "보건의료원", "보건소", "보건지소", "보건진료소", "약국"]

    data = merged_data[merged_data['시도'] != '전국']
    
    st.write('k값은 의료기관 방문환자수 * 의료기관 비율에 항생제 처방률을 곱하는 차수이다.')
    st.write('X = 의료기관 방문환자수 * 의료기관 비율 * (1/항생제 처방률)^k')
   
    k_input = st.text_input("직접 k 값을 입력하세요(0~10)", "2.5")  # k 값을 직접 입력 받음
    if k_input == "":
        k_in = 2.5
    else:
        k_in = min(max(float(k_input), 0.0), 10.0)

    # 사용자가 텍스트 상자에 k 값을 입력하면 슬라이더의 값을 해당 값으로 변경
    k = st.slider("k 값을 선택하세요", 0.0, 10.0, float(k_in))


    # 병원 종류 선택
    hos_type = st.selectbox("의료기관 종류 선택", hos_list)

    # "적용" 버튼 추가
    if st.button("적용"):
        # 데이터 처리 및 모델 적합
        data['x_adj'] = data[hos_type] * ((data['분모'] / data['분자']) ** k)
        X = data[['x_adj', '지표연도']]
        X = sm.add_constant(X)  # 상수항 추가
        y = data['CRE수']

        # 2중선형회귀 모델 적합
        model = sm.OLS(y, X).fit()

        # 예측값 생성
        merged_data['y_pred'] = model.predict(X)

        # 3D 시각화
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 산점도
        ax.scatter(data['x_adj'], data['지표연도'], data['CRE수'], color='blue', label='Actual Data')

        # 회귀면
        x1_surf, x2_surf = np.meshgrid(np.linspace(data['x_adj'].min(), data['x_adj'].max(), 100),
                                    np.linspace(data['지표연도'].min(), data['지표연도'].max(), 100))
        y_surf = model.params[0] + model.params[1] * x1_surf + model.params[2] * x2_surf
        ax.plot_surface(x1_surf, x2_surf, y_surf, color='red', alpha=0.5)

        # 축 라벨 및 제목 설정
        ax.set_xlabel('number of patient')
        ax.set_ylabel('YEAR')
        ax.set_zlabel('number of CRE')
        ax.set_title('3D plot of y vs x1 and x2')
        plt.legend()

        # 회귀 모델에서 회귀 계수 얻기
        beta_0 = model.params[0]
        beta_1 = model.params[1]
        beta_2 = model.params[2]

        # 예측 평면의 식
        predicted_surface_equation = f"y = {beta_0:.2f} + {beta_1:.2f} * x_1 + {beta_2:.2f} * x_2"

        st.write(f"회귀식 : y = {beta_0:.2f} + {beta_1:.2f} * x_1 + {beta_2:.2f} * x_2")

        # Streamlit에 그래프를 표시합니다
        st.pyplot(fig)

        # 결과 출력
        st.write(model.summary())

elif mnu == 'CRE 감염자 수 예측':
    import statsmodels.api as sm
    from mpl_toolkits.mplot3d import Axes3D

    data = merged_data[merged_data['시도'] != '전국']
    
    st.write('k값은 의료기관 방문환자수 * 의료기관 비율에 항생제 처방률을 곱하는 차수이다.')
    st.write('X = 의료기관 방문환자수 * 의료기관 비율 * (1/항생제 처방률)^k')
   
    k=2.625
    hos_type = '분자'

    # 사용자 입력: Year
    year_input = st.number_input("예측하고자 하는 년도을 입력하세요 (2021년 이후)", min_value=2021, value=2021, step=1)

    # 사용자 입력: 병원 방문자 수 범위
    visits = st.number_input("특정 기간 병원 방문자 수 입력", min_value=0, step=1, value=0)

    antibiotic = st.number_input("항생제 처방률 (0~1)", min_value=0.0, max_value=1.0, value=0.2)


    # "적용" 버튼 추가
    if st.button("적용"):
        # 데이터 처리 및 모델 적합
        data['x_adj'] = data[hos_type] * ((data['분모'] / data['분자']) ** k)
        X = data[['x_adj', '지표연도']]
        X = sm.add_constant(X)  # 상수항 추가
        y = data['CRE수']

        # 2중선형회귀 모델 적합
        model = sm.OLS(y, X).fit()

        # 예측값 생성
        merged_data['y_pred'] = model.predict(X)
        
        # 회귀 모델에서 회귀 계수 얻기
        beta_0 = model.params[0]
        beta_1 = model.params[1]
        beta_2 = model.params[2]

        y = beta_0 + beta_1 * visits * (1 / antibiotic ** k) + beta_2 * year_input

        if y<0:
            y = 0
        st.markdown(f'### {round(y)}')  