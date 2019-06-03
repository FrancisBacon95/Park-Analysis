# 필요한 패키지와 라이브러리를 가져옴
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

###################################
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False


park = pd.read_csv('C:\\Users\\최재혁\\Desktop\\컴시입\\data.csv', encoding='euc-kr', engine='python')
# OSError: Initializing from file failed라고 에러 발생해서 engine='python' 옵션을 추가해줌.

park.shape
#변수: 20개 
#관측치 : 16730개

#변수들의 타입 확인
park.dtypes
park['공원면적'].head()

#변수종류 확인 후 필요없는 것 제거
park.columns
park.drop(columns=['공원보유시설(운동시설)', '공원보유시설(유희시설)', '공원보유시설(편익시설)', '공원보유시설(교양시설)',
       '공원보유시설(기타시설)', '지정고시일', '관리기관명', '데이터기준일자','제공기관코드','제공기관명','Unnamed: 19'], inplace=True)
park.columns



park.head(10)
#결측치가 많음을 알 수 있음.

#데이터 내의 NA 개수를 찾아낸다.
park.isnull().sum()


msno.matrix(park)

#시각화를 했을 때,
#공원 크기에 따라 차이를 주기 위해 '공원면적비율' 이라는
#새로운 변수를 생성한다.
park['공원면적비율']=park['공원면적'].apply(lambda x : np.sqrt(x)*0.01)
park['공원면적비율']

 #도로명 주소 null값 확인
park['소재지도로명주소'].isnull().sum()
#8225개 확인

# 지번 주소의 널값 수
park['소재지지번주소'].isnull().sum() 
#854개 확인

msno.matrix(park)
#결측치를 시각화 시킨 이 그림을 보면
#대부분
#도로명 주소에서 결측치가 발생하면 지번 주소는 있고
#지번 주소에서 결측치가 발생하면 도로명 주소는 있다.

#그래서 둘 중에 하나라도 존재하면 그 하나로 채워준다.
park['소재지도로명주소'].fillna(park['소재지지번주소'], inplace= True )
park['소재지도로명주소'].isnull().sum()

#결측치 없는 것 확인

#지역별 공원분포를 보기 위해 도,광역시, 특별시 단위부터 해서 구,군 단위까지 좁혀나갈 수 있도록
#소재지 주소를 이용해서 새로운 변수들을 생성한다.
park['시도'] = park['소재지도로명주소'].str.split(' ',expand=True)[0]
park['구군'] = park['소재지도로명주소'].str.split(' ', expand=True)[1]

#위경도 시각화
(ggplot(park)
 + aes(x='경도', y='위도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)

#결측치 존재 확인
#데이터 전처리를 통해
#시도 결측치를 처리하거나
#아웃라이어 데이터를 제거하거나 대체를 해야한다.
#주소가 없는 관측치, 아웃라이어의 개수 확인과 제거 및 대체 실시
park_loc_notnull = park.loc[(park['위도'] > 32 ) & (park['경도'] < 130) & park['시도'].notnull()]
park.shape
park_loc_notnull.shape

# 위경도가 잘못입력된 데이터를 본다.
# 주소가 잘못되지는 않았다.
# 주소를 통해 위경도를 다시 받아온다.
outlier=park.loc[(park['위도'] < 30 ) | (park['경도'] >= 130)]
outlier["소재지도로명주소"]
park_loc_notnull["위도"][13304]=35.2123875
#사용할 데이터는 부산과 제주도이기 때문에 주소지가 부산인 데이터만
#수정했다.


#시도별 공원 데이터
(ggplot(park_loc_notnull)
 + aes(x='경도', y='위도', color='시도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)

#############################
####부산, 제주도만 남긴다.####
#############################

jeju = park_loc_notnull.loc[(park_loc_notnull['시도'] == "제주특별자치도") ]

# 전국적으로 어린이 공원이 가장 많은 것으로 보입니다.
# 제주도는 한라산 아래 해안선과 유사한 모습으로 공원이 배치되어 있는 모습이 인상적입니다.

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
    +aes(x="공원구분",y="공원면적비율")+geom_boxplot() + theme(text=element_text(family='Malgun Gothic')))



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


####랜덤포레스트####

'''
Y.loc[Y["Y"] == "근린공원", "Y"] = "N" 
Y.loc[Y["Y"] == "어린이공원", "Y"] = "K"
Y.loc[Y["Y"] == "체육공원", "Y"] = "A"
Y.loc[Y["Y"] == "문화공원", "Y"] = "C"
'''
#####################

jeju2 = jeju.loc[(jeju['공원구분'] == "근린공원")|
        (jeju['공원구분'] == "어린이공원")|    
        (jeju['공원구분'] == "체육공원")|
        (jeju['공원구분'] == "문화공원")]

#############
jeju2['Y']=jeju2['공원구분']
X=jeju2[['공원면적비율','위도','경도']]
Y=jeju2[['Y']]

Y.loc[ Y['Y'] == "근린공원", 'Y'] = 0
Y.loc[ Y['Y'] == "어린이공원", 'Y'] = 1
Y.loc[ Y['Y'] == "체육공원", 'Y'] = 2
Y.loc[ Y['Y'] == "문화공원", 'Y'] = 3
'''
Y.loc[Y["Y"] == "근린공원", "Y"] = "N" 
Y.loc[Y["Y"] == "어린이공원", "Y"] = "K"
Y.loc[Y["Y"] == "체육공원", "Y"] = "A"
Y.loc[Y["Y"] == "문화공원", "Y"] = "C"
'''
feature_names=["ratio of area","lat.","long."]

#셋분할
X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,test_size=0.3,random_state=0)



#트리모델 생성
#criterion='entropy' 분류 알고리즘 종류
seed=201514150
model=DecisionTreeClassifier(criterion='entropy',max_depth=4,random_state=seed)

model.fit(X_train,Y_train)
print("학습용 데이터 정확도 : {:.3f}".format(model.score(X_train,Y_train)))
print("검증용 데이터 정확도 : {:.3f}".format(model.score(X_test, Y_test)))





#경도 : long. 위도:lat.
dot_data=export_graphviz(model,out_file=None, feature_names=feature_names,
                         rounded=True,class_names=["Neighbor","Kids","Athletic","Culture"], special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())