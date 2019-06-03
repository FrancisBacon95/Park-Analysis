# 본인의 과제명 작성

학과 | 학번 | 성명
---- | ---- | ---- 
통계학과 |201514150 |최재혁


## 프로젝트 개요
1. 데이터 소개 및 변수설명.
2. EDA(탐색적자료 분석)
	1) 아웃라이어 및 결측치 처리
	2) 데이터 시각화
3. 데이터 분석
4. Decision Tree Model을 이용한 분석 결과 검증
5. 결론 

## 사용한 공공데이터 
[데이터보기](https://github.com/cybermin/python2019/blob/master/%EB%B6%80%EC%82%B0%EA%B5%90%ED%86%B5%EA%B3%B5%EC%82%AC_%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EC%97%AD%EC%82%AC%EC%A0%95%EB%B3%B4_20190520.csv)

## 소스
* [링크로 소스 내용 보기]
(https://github.com/cybermin/python2019/blob/master/tes.py) 

* 필요한 라이브러리를 가져옵니다.

~~~python
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
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
~~~
*데이터 확인
~~~python
park = pd.read_csv('C:\\Users\\최재혁\\Desktop\\컴시입\\data.csv', encoding='euc-kr', engine='python')
# OSError: Initializing from file failed라고 에러 발생해서 engine='python' 옵션을 추가해줌.

park.shape
park.head(5)
~~~

*변수 설명 및 타입 확인
~~~python
#변수들의 타입 확인
park.dtypes
~~~

*변수 제거

*주제와 관련 없는 변수들을 제거합니다.
~~~python
park.columns
park.drop(columns=['공원보유시설(운동시설)', '공원보유시설(유희시설)', '공원보유시설(편익시설)', '공원보유시설(교양시설)',
       '공원보유시설(기타시설)', '지정고시일', '관리기관명', '데이터기준일자','제공기관코드','제공기관명','Unnamed: 19'], inplace=True)
park.columns
~~~

*결측치 파악

*missingno 패키지를 이용하여 결측치의 존재를 시각적으로 확인합니다.

~~~python
park.isnull().sum()
msno.matrix(park)
~~~
*'소재지도로명주소'와 '소재지지번주소' 변수 중에서 결측치가 존재하더라도 

둘 중의 하나는 관측값이 있으므로 서로를 채워주면 결측치를 처리할 수 있습니다.

~~~python
park['소재지도로명주소'].fillna(park['소재지지번주소'], inplace= True )
park['소재지도로명주소'].isnull().sum()
msno.matrix(park)
~~~

특별시/도/광역시로 구분하고 시/구/군으로 구분하기 위해

주소지를 분할하여 새로운 변수를 만들어준다.
~~~python
park['시도'] = park['소재지도로명주소'].str.split(' ',expand=True)[0]
park['구군'] = park['소재지도로명주소'].str.split(' ', expand=True)[1]
~~~

위/경도 데이터 확인

위/경도에 대하여 outlier는 없는지 확인하기 위해

ggplot을 이용해 scatter plot을 그려 확인합니다.

~~~python
(ggplot(park_loc_notnull)
 + aes(x='경도', y='위도', color='시도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)
~~~

국내 공원에 대한 데이터임에도 불구하고 국외의 위치에 찍혀있는

outlier를 확인할 수 있습니다.

이것들을 제거해줄지 수정해줄지를 확인하기 위해

주소지를 확인 해봅니다.

~~~python
outlier=park.loc[(park['위도'] < 30 ) | (park['경도'] >= 130)]
outlier["소재지도로명주소"]
park_loc_notnull["위도"][13304]=35.2123875
~~~
주소지의 경우는 국내로 제대로 되어 있으나 

위/경도가 잘못 표시된 것으로 판단했습니다.

그러나 이 관측치들의 경우에는 이번 분석에 사용될 관측치들이 아니기 때문에

부산에 있는 데이터만 수정을 해준 뒤

나머지는 제거합니다.

~~~python
park_loc_notnull = park.loc[(park['위도'] > 32 ) & (park['경도'] < 130) & park['시도'].notnull()]
park.shape
park_loc_notnull.shape
~~~

이후에 분석을 위해 시각화를 했을 때,

공원 크기에 따라 차이를 주기 위해 

'공원면적비율' 이라는 새로운 변수를 생성한다.
~~~python
park['공원면적비율']=park['공원면적'].apply(lambda x : np.sqrt(x)*0.01)
park['공원면적비율']
~~~

이것으로 분석에 필요한 데이터 전처리는 모두 완료했습니다.

분석에 들어가기에 앞서 전국의 공원 데이터를 특별시/도/광역시로 구분하여

전체적인 분포를 간단히 살펴보겠습니다.

~~~python
(ggplot(park_loc_notnull)
 + aes(x='경도', y='위도', color='시도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)
~~~

# ANALYSIS

전처리가 끝난 데이터를 제주도에 해당되는 데이터만 다시 선별하여

분석을 시작합니다.

~~~python
jeju = park_loc_notnull.loc[(park_loc_notnull['시도'] == "제주특별자치도") ]
~~~


### 공원구분별 합계 확인
~~~python
jeju_park = jeju['공원구분'].value_counts().reset_index()
jeju_park.columns = ('공원구분', '합계')
jeju_park

(ggplot(jeju_park)
 + aes(x='공원구분', y='합계')
 + geom_bar(stat='identity', position='dodge', fill='green')
 + coord_flip()
 + theme(text=element_text(family='Malgun Gothic'))
)
~~~

### 공원구분별 분포 확인

제주도 내의 공원들을 종류별로 구분하여 

시각화를 통해 분포를 확인합니다.


~~~python
(ggplot(jeju)
 + aes(x='경도', y='위도', color='공원구분', size='공원면적비율') 
 + geom_point()
 + geom_jitter(color='lightgray', alpha=0.25)
 + theme(text=element_text(family='Malgun Gothic'))
)
~~~
1) 공원들이 주로 제주도의 중심부보다는 

  해안선을 따라 많이 분포되어있음을 알 수 있습니다.

2) 근린공원의 경우에는 전체적으로 넓게 분포되어 있는 형태이고
  
  어린이공원의 경우에는 북쪽, 남쪽에 밀집되어 있는 형태임을 알 수 있습니다.
  그 외의 공원의 경우에는 공원의 개수 자체가 매우 적기 때문에 분포를 파악하기가 부적절 합니다.

3) 공원면적비율이 '근린공원'이 주로 큰 것을 위의 그림을 통해

  확인할 수 있었습니다.

  이를 확인하기 위해 boxplot을 이용합니다.

## 공원구분별 면적비율 확인
~~~python
(ggplot(jeju)
    + aes(x="공원구분",y="공원면적비율")
    + geom_boxplot() 
    + theme(text=element_text(family='Malgun Gothic')))
~~~
1) 근린공원의 경우는 전체적으로 면적비율이 큰 편임을 알 수 있다.

2) 어린이공원의 경우는 면적비율이 다른 공원들에 비해 매우 작음을 알 수 있다.

3) 그 외의 공원들의 경우에는 관측치 개수가 매우 적기 때문에 
  
  구분별 크기의 성질을 파악하는 것은 부적절하다.
  
## 제주시, 서귀포시 확대하여 중심부 일부만을 확인
~~~python
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
~~~
지금까지의 분석을 확인하기 위해 밀집지역을 확대하여 살펴봅니다.


제주시의 경우에는 공항근처에 공원이 많이 밀집되어 있고,

서귀포시의 경우에는 중문 근처에 공원이 많이 밀집되어 있음을 확인할 수 있습니다.
   
이를 통해 주로 사람들이 많이 몰리는 곳에 공원 또한 많이 분포되어 있음을 알 수 있습니다.
