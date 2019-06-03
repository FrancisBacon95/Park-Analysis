import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from plotnine import *
import re
import folium
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
from IPython.display import Image
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

park = pd.read_csv('C:\\Users\\최재혁\\Desktop\\컴시입\\data.csv', encoding='euc-kr', engine='python')
# OSError: Initializing from file failed라고 에러 발생해서 engine='python' 옵션을 추가해줌.

park.shape
park.head(5)

#변수들의 타입 확인
park.dtypes


park.columns
park.drop(columns=['공원보유시설(운동시설)', '공원보유시설(유희시설)', '공원보유시설(편익시설)', '공원보유시설(교양시설)',
       '공원보유시설(기타시설)', '지정고시일', '관리기관명', '데이터기준일자','제공기관코드','제공기관명','Unnamed: 19'], inplace=True)
park.columns

park.isnull().sum()
msno.matrix(park)

park['소재지도로명주소'].fillna(park['소재지지번주소'], inplace= True )
park['소재지도로명주소'].isnull().sum()
msno.matrix(park)

park['시도'] = park['소재지도로명주소'].str.split(' ',expand=True)[0]
park['구군'] = park['소재지도로명주소'].str.split(' ', expand=True)[1]
park[['시도']].head(5)
park[['구군']].head(5)

(ggplot(park)
 + aes(x='경도', y='위도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)


outlier=park.loc[(park['위도'] < 30 ) | (park['경도'] >= 130)]
outlier["소재지도로명주소"]
park["위도"][13304]=35.2123875

park_loc_notnull = park.loc[(park['위도'] > 32 ) & (park['경도'] < 130) & park['시도'].notnull()]
park.shape
park_loc_notnull.shape

park_loc_notnull['공원면적비율']=park_loc_notnull['공원면적'].apply(lambda x : np.sqrt(x)*0.01)
park_loc_notnull['공원면적비율']

(ggplot(park_loc_notnull)
 + aes(x='경도', y='위도', color='시도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)








jeju = park_loc_notnull.loc[(park_loc_notnull['시도'] == "제주특별자치도") ]

jeju_park = jeju['공원구분'].value_counts().reset_index()
jeju_park.columns = ('공원구분', '합계')
jeju_park

(ggplot(jeju_park)
 + aes(x='공원구분', y='합계')
 + geom_bar(stat='identity', position='dodge', fill='green')
 + coord_flip()
 + theme(text=element_text(family='Malgun Gothic'))
)

(ggplot(jeju)
 + aes(x='경도', y='위도', color='공원구분', size='공원면적비율') 
 + geom_point()
 + geom_jitter(color='lightgray', alpha=0.25)
 + theme(text=element_text(family='Malgun Gothic'))
)

(ggplot(jeju)
    + aes(x="공원구분",y="공원면적비율")
    + geom_boxplot() 
    + theme(text=element_text(family='Malgun Gothic')))

jeju_jeju = jeju.loc[jeju['구군'] == '제주시']
(ggplot(jeju_jeju)
 + aes(x='경도', y='위도', color='공원구분', size='공원면적비율') 
 + geom_point()
 + geom_jitter(color='lightgray', alpha=0.25)
 + theme(text=element_text(family='Malgun Gothic')) +xlim(126.3,126.7))

jeju_jeju = jeju.loc[jeju['구군'] == '서귀포시']
(ggplot(jeju_jeju)
 + aes(x='경도', y='위도', color='공원구분', size='공원면적비율') 
 + geom_point()
 + geom_jitter(color='lightgray', alpha=0.25)
 + theme(text=element_text(family='Malgun Gothic')) +xlim(126.4,126.8)+ylim(33.2,33.3))

jeju2 = jeju.loc[(jeju['공원구분'] == "근린공원")|
        (jeju['공원구분'] == "어린이공원")|    
        (jeju['공원구분'] == "체육공원")|
        (jeju['공원구분'] == "문화공원")]


jeju2['Y']=jeju2['공원구분']
X=jeju2[['공원면적비율','위도','경도']]
Y=jeju2[['Y']]

Y.loc[ Y['Y'] == "근린공원", 'Y'] = 0
Y.loc[ Y['Y'] == "어린이공원", 'Y'] = 1
Y.loc[ Y['Y'] == "체육공원", 'Y'] = 2
Y.loc[ Y['Y'] == "문화공원", 'Y'] = 3

feature_names=["ratio of area","lat.","long."]

X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,test_size=0.3,random_state=0)

seed=201514150
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=seed)

model.fit(X_train,Y_train)
print("학습용 데이터 정확도 : {:.3f}".format(model.score(X_train,Y_train)))
print("검증용 데이터 정확도 : {:.3f}".format(model.score(X_test, Y_test)))

#경도 : long. 위도:lat.
dot_data=export_graphviz(model,out_file=None, feature_names=feature_names,
                         rounded=True,class_names=["Neighbor","Kids","Athletic","Culture"], special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())