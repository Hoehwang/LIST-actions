# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import pandas as pd

import numpy as np
import re
import random
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

res = pd.read_csv("./actions/RESPONSE_EXP_FIN.csv", encoding = 'utf8')
syn = pd.read_csv("./actions/SYN.csv", encoding = 'utf8')
rec_ta = pd.read_csv('./actions/recommend_table_list.csv', encoding = 'utf8')

#int, ent mapper
int_ent_mapper = {}
with open('./actions/INT_ENT_MAPPER.txt', 'r', encoding = 'utf8') as f:
    data = f.readlines()
    data = list(map(lambda x: x.strip(), data))
for d in data:
    key, value = d.split('\t')
    int_ent_mapper[key] = value.split(', ') # value = [column, value]

#default location, accom
default_location = 'GANGNAM'
default_accom = 'HOTEL'

default_tag_changer = {"<LOCATION-TYPE_FEATURE>": "해당 지역",
                        "<ACCOMMODATION-TYPE_FEATURE>": "숙소",
                        "<PROVIDED_FEATURE>": "특정 비품",
                        "<SURROUND_FEATURE>": "말씀하신 시설",
                        "<NATURE_FEATURE>": "자연",
                        "<BAN_FEATURE>": "금지 사항",
                        "<GOOD-TO-GO_FEATURE>": "말씀하신 목적으로"}

class ActionRephraseResponse(Action):

    def __init__(self):
        self.no_result = 0
        self.accom_type_changed = 0
        self.city_type_changed = 0
        self.mapped_entity_ls = []
    # 액션에 대한 이름을 설정
    def name(self) -> Text:
        return "action_rephrase_accommodation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        self.no_result = 0
        self.accom_type_changed = 0
        self.city_type_changed = 0
        self.mapped_entity_ls = []

        print(tracker.latest_message["entities"])
        print(tracker.get_intent_of_latest_message())

        entity = [a.get("entity") for a in tracker.latest_message["entities"]]
        intent = tracker.get_intent_of_latest_message()

        sort_key, find_value = self.map_int_ent(int_ent_mapper, entity, intent)
        sort_key += ['rank']
        print(sort_key)
        print(find_value)
        print(self.no_result, self.accom_type_changed, self.city_type_changed)

        rec_ta_sorted_by_values = rec_ta.sort_values(by=sort_key ,ascending=False)
        output_img_link = self.apply_categorical_condition(rec_ta_sorted_by_values, find_value)


        #답변하기, intent에 따라 답변이 다름
        #all-feature, 현재는 다른 intent와 처리가 같으나 추후에 바뀔 수 있음

        if intent.lower() == 'all-feature': 
            ## 데이터가 없는 경우 - 데이터가 없다고 말하기
            if self.no_result == 1:
                dispatcher.utter_message(text=res[res["intent"]==intent]["entityless"].values[0])
            else:
                ## 그 외 아래와 같이 진행
                # NLG 폼 가져오기, 여러 개 중 하나 선택
                nlg_form = random.choice(res[res["intent"]==intent]["response"].values[0].split(' / '))
                # SLOT 찾기
                slots = re.findall('<.+?>', nlg_form)
                slots = list(map(lambda x: x.replace('_FEATURE>', '').lstrip('<'), slots))
                # 해당 SLOT에 들어갈 SYN 찾기(엔티티 중에서), SLOT과 SYN 치환
                for slot in slots:
                    slot_changed = False
                    for ent in self.mapped_entity_ls:
                        temp = syn[(syn["position"]==slot) & (syn['entity-name']==ent.upper())]['norm'].values
                        if temp:
                            nlg_form = nlg_form.replace('<'+slot+'_FEATURE>', temp[0])
                            slot_changed = True
                            break
                    if slot_changed == False and '<'+slot+'_FEATURE>' in list(default_tag_changer.keys()):
                        if slot == 'LOCATION-TYPE':
                            nlg_form = re.sub('<'+slot+'_FEATURE>\S* ?', '', nlg_form)
                        else:
                            nlg_form = nlg_form.replace('<'+slot+'_FEATURE>', default_tag_changer['<'+slot+'_FEATURE>'])

                # 답변하기
                dispatcher.utter_message(text=nlg_form)
                dispatcher.utter_message(text=res[res["intent"]==intent]["send_link"].values[0])
                #링크 답변(사진, 링크)
                num = 1
                for img, link in output_img_link:
                    text = str(num) + '위 숙소) ' + ': ' + link
                    dispatcher.utter_message(text = text)
                    dispatcher.utter_message(image=img)
                    num += 1
                dispatcher.utter_message(text=res[res["intent"]==intent]["utter_ask_more"].values[0])
        else:
            ## 데이터가 없는 경우 - 데이터가 없다고 말하기
            if self.no_result == 1:
                dispatcher.utter_message(text=res[res["intent"]==intent]["entityless"].values[0])
                num = 1
                for img, link in output_img_link:
                    text = str(num) + '위 숙소) ' + ': ' + link
                    dispatcher.utter_message(text = text)
                    dispatcher.utter_message(image=img)
                    num += 1
            else:
                ## 그 외 아래와 같이 진행
                # NLG 폼 가져오기, 3개 중 하나 선택
                nlg_form = random.choice(res[res["intent"]==intent]["response"].values[0].split(' / '))
                # SLOT 찾기
                slots = re.findall('<.+?>', nlg_form)
                slots = list(map(lambda x: x.replace('_FEATURE>', '').lstrip('<'), slots))
                # 해당 SLOT에 들어갈 SYN 찾기(엔티티 중에서), SLOT과 SYN 치환
                for slot in slots:
                    slot_changed = False
                    for ent in self.mapped_entity_ls:
                        temp = syn[(syn["position"]==slot) & (syn['entity-name']==ent.upper())]['norm'].values
                        if temp:
                            slot_changed = True
                            nlg_form = nlg_form.replace('<'+slot+'_FEATURE>', temp[0])
                            break
                    if slot_changed == False and '<'+slot+'_FEATURE>' in list(default_tag_changer.keys()):
                        if slot == 'LOCATION-TYPE':
                            nlg_form = re.sub('<'+slot+'_FEATURE>\S* ?', '', nlg_form)
                        else:
                            nlg_form = nlg_form.replace('<'+slot+'_FEATURE>', default_tag_changer['<'+slot+'_FEATURE>'])
                    
                # 답변하기
                dispatcher.utter_message(text=nlg_form)
                dispatcher.utter_message(text=res[res["intent"]==intent]["send_link"].values[0])

                #링크 답변(사진, 링크)
                num = 1
                for img, link in output_img_link:
                    text = str(num) + '위 숙소) ' + ': ' + link
                    dispatcher.utter_message(text = text)
                    dispatcher.utter_message(image=img)
                    num += 1

                dispatcher.utter_message(text=res[res["intent"]==intent]["utter_ask_more"].values[0])

    def map_int_ent(self, int_ent_mapper, entity, intent):
        #entity_mapping, intent도 추가
        #MAPPER의 유형
        #1) sort_key : COL_NAME, NONE -> COL_NAME으로 정렬(인텐트와 엔티티 모두)
        #2) 무시 : NONE, INTENT -> INTENT 중 쓸 데 없는 것
        #3) find_value : dict 형식,  COL_NAME, VALUE -> COL_NAME 중 VALUE 찾기, categorical value, domain마다 상이

        sort_key = []
        find_value = {'cate': '', 'sub-category': '', 'city' : '', 'city-onto' : ''}

        for e in entity:
            temp = int_ent_mapper[e]
            if temp[0] != 'NONE':
                if temp[1] == 'NONE':
                    sort_key.append(temp[0].lower())
                    self.mapped_entity_ls.append(temp[0])
                else:
                    find_value[temp[0].lower()] = temp[1]
                    self.mapped_entity_ls.append(temp[1])
        

        temp = int_ent_mapper[intent]
        if temp[0] != 'NONE':
            if temp[1] == 'NONE':
                sort_key.append(temp[0].lower())
            else:
                find_value[temp[0].lower()] = temp[1]
                
        
        #city값 못 잡았거나 없었으면 default 값 적용
        # if find_value['city'] == '':
        #     find_value['city'] = default_location
        #     self.mapped_entity_ls.append(default_location)

        return list(set(sort_key)), find_value

    def apply_categorical_condition(self, df, find_value):
        #출력할 이미지, 숙소 이름, 링크
        output_img_name_link = []

        #주어진 df로부터 find_value form 속 정보를 활용하여 categorical한 value 조건을 적용
        res_df = df
        for col_name in list(find_value.keys()):
            if find_value[col_name] != '':
                res_df = res_df[res_df[col_name] == find_value[col_name]]
        
        if res_df.size == 0: #조건에 맞는 숙소가 없으면(보통 지역-숙소 조합이 문제)
            self.accom_type_changed = 1
            res_df = df
            find_value['cate'] = '' # 숙박 유형을 all로 하고 찾기
            for col_name in list(find_value.keys()):
                if find_value[col_name] != '':
                    res_df = res_df[res_df[col_name] == find_value[col_name]]
        
        if res_df.size == 0: #숙박 유형 전체에도 조건에 맞는 숙소가 없으면(보통 지역-숙소 조합이 문제)
            self.city_type_changed = 1
            res_df = df
            find_value['city-onto'] = rec_ta[rec_ta['city'] == find_value['city']]['city-onto'].to_list()[0] # 현재 도시의 상위 도시로 색인
            find_value['city'] = ''
            for col_name in list(find_value.keys()):
                if find_value[col_name] != '':
                    res_df = res_df[res_df[col_name] == find_value[col_name]]

        if res_df.size == 0: #그래도 없으면,
            self.no_result = 1
            output_img_name_link = list(zip(df['image'].to_list()[:3], df['url'].to_list()[:3]))
        elif res_df.size < 3:
            output_img_name_link = list(zip(res_df['image'].to_list()[:res_df.size], res_df['url'].to_list()[:res_df.size]))
        else:
            output_img_name_link = list(zip(res_df['image'].to_list()[:3], res_df['url'].to_list()[:3]))


        return output_img_name_link
    
    #들어온 값들 (INTENT, ENTITY)-FEATURE로 숙소 탐색, INTENT, ENTITY MAPPER(NLU의 전체 INTENT, ENTITY를 보고 TABLE이랑 매칭 시켜줌)
    #ENTITY_MAPPER: INTENT (TYPE, VALUE): TYPE은 참조하는 열이름, VALUE는 실제 찾을 값, ENTITY의 경우, 경우에 따라 TYPE과 VALUE가 동일, 혹은 VALUE를 NONE으로
    #ENTITY 중 지역, 숙박유형/세부 숙박유형 골라냄
    #지역 DEFAULT = 서울
    #FEATURE -> RANK 순으로 정렬(내림차순, COUNT가 0이어도 나오게), 지역=ENTITY, 숙박유형=ENTITY로 고정, 상위 3개 추출, 지역이나 숙박 유형이 없을 수 있음
    #1) 숙박유형이 없을 경우 -> 숙박유형 고정 풀고 처리
    #2) 지역이 없을 경우 -> 도시 상위 -> 없으면 서울
    
    #뽑은 애들 중 1등을 검사해서 조건에 모두 안 맞으면(모두 0이면, ) 조건에 맞는 숙소가 없다고 말하면서 추천
