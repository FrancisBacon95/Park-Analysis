# 제주도의 공원 분포 분석

학과 | 학번 | 성명
---- | ---- | ---- 
통계학과 |201514150 |최재혁

## 프로젝트 목적 및 동기
 학기가 끝나고 방학이 오면 많은 사람들이 여행을 떠납니다.
 
 우리나라의 가족여행지로써 대표적인 관광지를 뽑자면 제주도를 빼놓을 수 없습니다. 
 
 
 실제로 제주도 내에는 근린공원, 역사공원, 어린이공원 등 여러 종류의 공원들이 존재합니다. 
 
 그래서 저는 이번 프로젝트를 통해 제주도에는 어떤 공원들이 주로 존재하며, 
 
 그것들은 주로 어디에 위치해있고 그들의 분포는 어떻게 되는지에 대해서 
 
 시각화를 통해 쉽게 식별할 수 있는 프로그램을 만들어보았습니다. 
 
 
## 프로젝트 개요
1. 데이터 소개 및 변수설명.
2. EDA(탐색적자료 분석)
	1) 아웃라이어 및 결측치 처리
	2) 데이터 시각화
3. 데이터 분석
4. Decision Tree Model을 이용한 분석 결과 검증
5. 결론 

## 사용한 공공데이터 
[데이터보기](https://www.data.go.kr/dataset/15012890/standard.do)

## 소스
* [링크로 소스 내용 보기](https://github.com/michinnom/-/blob/master/201514150최재혁(Py코드)수정.py) 




## 데이터 확인
~~~python
park = pd.read_csv('C:\\Users\\최재혁\\Desktop\\컴시입\\data.csv', encoding='euc-kr', engine='python')
# OSError: Initializing from file failed라고 에러 발생해서 engine='python' 옵션을 추가해줌.

park.shape
park.head(5)
~~~
![01](https://user-images.githubusercontent.com/51112316/58788418-d7dd6200-8626-11e9-9d51-e03512cb7aaf.JPG)

## 변수 설명 및 타입 확인
### 변수설명은 직관적으로 해석가능하기에 생략한다.
~~~python
#변수들의 타입 확인
park.dtypes
~~~
![02](https://user-images.githubusercontent.com/51112316/58788419-d7dd6200-8626-11e9-8d1f-6ae32911899d.JPG)


## 필요한 패키지와 라이브러리를 가져옴

~~~python
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

## 변수 제거

주제와 관련 없는 변수들을 제거합니다.
~~~python
park.columns
park.drop(columns=['공원보유시설(운동시설)', '공원보유시설(유희시설)', '공원보유시설(편익시설)', '공원보유시설(교양시설)',
       '공원보유시설(기타시설)', '지정고시일', '관리기관명', '데이터기준일자','제공기관코드','제공기관명','Unnamed: 19'], inplace=True)
park.columns
~~~
![03](https://user-images.githubusercontent.com/51112316/58788420-d875f880-8626-11e9-9747-5b13bf1e3df2.JPG)

## 결측치 파악

missingno 패키지를 이용하여 결측치의 존재를 시각적으로 확인합니다.

~~~python
park.isnull().sum()
msno.matrix(park)
~~~
![04_1](https://user-images.githubusercontent.com/51112316/58788421-d875f880-8626-11e9-88fa-cca574722d1e.JPG)
![04_2](https://user-images.githubusercontent.com/51112316/58788422-d90e8f00-8626-11e9-95a0-ac1ad8380be5.png)
'소재지도로명주소'와 '소재지지번주소' 변수 중에서 결측치가 존재하더라도 

둘 중의 하나는 관측값이 있으므로 서로를 채워주면 결측치를 처리할 수 있습니다.

~~~python
park['소재지도로명주소'].fillna(park['소재지지번주소'], inplace= True )
park['소재지도로명주소'].isnull().sum()
msno.matrix(park)
~~~
![05_1](https://user-images.githubusercontent.com/51112316/58788424-d90e8f00-8626-11e9-95af-8919b9beb255.JPG)
![05_2](https://user-images.githubusercontent.com/51112316/58788425-d90e8f00-8626-11e9-90f2-e34d4ddaf68e.png)

특별시/도/광역시로 구분하고 시/구/군으로 구분하기 위해

주소지를 분할하여 새로운 변수를 만들어준다.
~~~python
park['시도'] = park['소재지도로명주소'].str.split(' ',expand=True)[0]
park['구군'] = park['소재지도로명주소'].str.split(' ', expand=True)[1]
~~~
![06](https://user-images.githubusercontent.com/51112316/58788426-d9a72580-8626-11e9-8e55-d87840f4a3b8.JPG)

위/경도 데이터 확인

위/경도에 대하여 outlier는 없는지 확인하기 위해

ggplot을 이용해 scatter plot을 그려 확인합니다.

~~~python
(ggplot(park)
 + aes(x='경도', y='위도')
 + geom_point()
 + theme(text=element_text(family='Malgun Gothic'))
)
~~~
![07](https://user-images.githubusercontent.com/51112316/58788428-d9a72580-8626-11e9-88f9-fa4a4c255208.png)

국내 공원에 대한 데이터임에도 불구하고 국외의 위치에 찍혀있는

outlier를 확인할 수 있습니다.

이것들을 제거해줄지 수정해줄지를 확인하기 위해

주소지를 확인 해봅니다.

~~~python
outlier=park.loc[(park['위도'] < 30 ) | (park['경도'] >= 130)]
outlier["소재지도로명주소"]
park["위도"][13304]=35.2123875
~~~
![08](https://user-images.githubusercontent.com/51112316/58788429-d9a72580-8626-11e9-9319-74349e95fce1.JPG)

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
![09](https://user-images.githubusercontent.com/51112316/58788431-da3fbc00-8626-11e9-9061-c7f2cad9367e.JPG)

이후에 분석을 위해 시각화를 했을 때,

공원 크기에 따라 차이를 주기 위해 

'공원면적비율' 이라는 새로운 변수를 생성한다.
~~~python
park_loc_notnull['공원면적비율']=park_loc_notnull['공원면적'].apply(lambda x : np.sqrt(x)*0.01)
park_loc_notnull['공원면적비율']
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
![10](https://user-images.githubusercontent.com/51112316/58788432-da3fbc00-8626-11e9-9220-28089982d8b2.png)



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
![11_01](https://user-images.githubusercontent.com/51112316/58788433-da3fbc00-8626-11e9-997b-07039e92824a.JPG)
![11-02](https://user-images.githubusercontent.com/51112316/58788434-da3fbc00-8626-11e9-9683-968942923086.png)


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
![12](https://user-images.githubusercontent.com/51112316/58788435-dad85280-8626-11e9-90e4-7a06fb31fef4.png)


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
![13](https://user-images.githubusercontent.com/51112316/58788436-dad85280-8626-11e9-9306-699e640fcdd8.png)

1) 근린공원의 경우는 전체적으로 면적비율이 큰 편임을 알 수 있다.

2) 어린이공원의 경우는 면적비율이 다른 공원들에 비해 매우 작음을 알 수 있다.

3) 그 외의 공원들의 경우에는 관측치 개수가 매우 적기 때문에 
  
    구분별 크기의 성질을 파악하는 것은 부적절하다.
  
  
## 제주시, 서귀포시 확대하여 중심부 일부만을 확인

지금까지의 분석을 확인하기 위해 밀집지역을 확대하여 살펴봅니다.

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
![14_01](https://user-images.githubusercontent.com/51112316/58788437-dad85280-8626-11e9-982a-2f7ae23e98d6.png)
![14-02](https://user-images.githubusercontent.com/51112316/58788438-db70e900-8626-11e9-8c8c-b735b28491bd.png)

제주시의 경우에는 공항근처에 공원이 많이 밀집되어 있고,

서귀포시의 경우에는 중문 근처에 공원이 많이 밀집되어 있음을 확인할 수 있습니다.
   
이를 통해 주로 사람들이 많이 몰리는 곳에 공원 또한 많이 분포되어 있음을 알 수 있습니다.


# 결론

* 제주도에는 공원이 해안선을 따라 분포되어 있다.

* 제주도에는 어린이공원과 근린공원이 대부분이다.

* 근린공원의 경우는 규모가 큰 편이나 어린이공원의 경우는 규모가 작은 편이다.

* 어린이공원은 "제주시 : 제주공항", "서귀포시 : 중문지역"에 공원이 밀집되어 있다.

* 근린공원은 제주도 전반에 걸쳐 넓게 분포되어 있다.


# 검증

## DECISISON TREE(의사결정나무)모델을 통한 검증

의사결정나무 모델을 생성하여 위의 제가 내린 결론들을 대략적으로 검증해보았습니다.

의사결정나무는 기계 학습(Machine Learning)에서 사용하는 모델링 방법 중 하나입니다.

예측, 분류 등 여러 분야로 사용되고 응용될 수 있는데 이번 프로젝트 목적 상

간단히 분류하여 그 분류모델을 통해 제 분석이 타당한지를 검증해보는 용도로 사용했습니다.

### 의사결정나무

이번 프로젝트에 사용된 의사결정나무 기법에 대해 간단히 설명하자면

각 설명변수에 변수중요도를 매겨 변수중유도에 따라 데이터를 분류합니다.

데이터에 대하여 A1 조건을 주고 "참이면, B1조건으로.", "거짓이면, B2조건으로." 

이처럼 이분법적으로 나누고 그 속에서 또 다시 

"B1 조건이 참이면, C1조건으로.", "B1조건이 거짓이면, C2조건으로.", 

"B2조건이 참이면, C3조건으로.", "B2조건이 거짓이면, C4조건으로."

......

이렇게 계속적으로 뻗어 내려가 데이터를 분류해내는 모델입니다.

## 모델링을 위한 데이터 전처리

반응변수(Y) : 공원구분
설명변수(X1, X2, X3) : 공원면적비율, 위도, 경도 

~~~python
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

feature_names=["ratio of area","lat.","long."]
~~~
### TRAIN SET, TEST SET으로 분할 (7:3)
원래는 TRAIN SET, VALIDATION SET, TEST SET으로 나누어 

VALIDATION SET으로 모델 성능을 평가하면서 올린 후에 

이후에 제공되는 데이터인 TEST SET으로 학습된 모델을 평가 받는 것입니다.

(비유 하자면 TRAIN : 교과서, VALIDATION : 퀴즈, TEST : 시험)  

그런데 이번 프로젝트 목적상 좋은 모델을 만드는 게 아니라

EDA를 통해 제가 얻은 분석을 검증을 하기 위함이므로 

간략하게 전체데이터를 7:3 = TRAIN : TEST 으로 분할하여

TRAIN SET으로 모델을 만들어 그 모델을 TEST SET의 데이터를 검증하는 방식으로 수행했습니다.

~~~python
X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y,test_size=0.3,random_state=0)
~~~

## 모델
~~~python
seed=201514150
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=seed)

model.fit(X_train,Y_train)
print("학습용 데이터 정확도 : {:.3f}".format(model.score(X_train,Y_train)))
print("검증용 데이터 정확도 : {:.3f}".format(model.score(X_test, Y_test)))
~~~
![15](https://user-images.githubusercontent.com/51112316/58788440-dc097f80-8626-11e9-80f9-e695ad275cbb.JPG)

## 모델 시각화
~~~python
#경도 : long. 위도:lat.
dot_data=export_graphviz(model,out_file=None, feature_names=feature_names,
                         rounded=True,class_names=["Neighbor","Kids","Athletic","Culture"], special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
~~~
![16](https://user-images.githubusercontent.com/51112316/58788441-dc097f80-8626-11e9-9f1b-92eac810ff4b.png)

1) 변수중요도 : 공원면적비율 >> 위도(lat.) >>> 경도(long.)
    임을 시각화를 통해 대략적으로 식별할 수 있다.

2) 사실상 공원을 나누는 기준은 면적이라고 볼 수 있다.
    
3) 주로 근린공원이 공원면적비율이 큼을 알 수 있다. 
    (체육공원도 매우 크지만 관측치 개수가 매우 적으므로 그 이상의 해석이 불가능하다.)
    
4) 그 이상의 해석은 데이터 불균형으로 인해 불가능하므로 

    Oversampling, Undersampling, SMOTE(Synthetic Minority Over-sampling Technique) 
    
    등의 기법을 추가적으로 사용해야 한다. 

# 최종결론

* 공원을 나누는 기준은 면적이다.

* 제주도에는 공원이 해안선을 따라 분포되어 있다.

* 제주도에는 어린이공원과 근린공원이 대부분이다.

* 근린공원의 경우는 규모가 큰 편이나 어린이공원의 경우는 규모가 작은 편이다.

* 어린이공원은 "제주시 : 제주공항", "서귀포시 : 중문지역"에 공원이 밀집되어 있다.

* 근린공원은 제주도 전반에 걸쳐 넓게 분포되어 있다.

