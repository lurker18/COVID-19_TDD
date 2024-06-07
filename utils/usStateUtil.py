import json
import pandas as pd

def getRegionNames():
    states = json.load(open('./data/us_adjacent_states.json', 'r'))
    states["US"] = []
    
    return list(states.keys())
#end def

def getRegionCodes():
    states = json.load(open('./data/us_adjacent_states.json', 'r'))
    states["US"] = []
    state_codes = []
    for state in list(states.keys()):
        state_codes.append(convRegionName2Code(state))
    #end for
    return state_codes
#end def
        

# name > code
def convRegionName2Code(region_name):
    ''' 지역 이름을 코드로 변환하는 함수
    Args:
      region_name (str): 지역의 이름 ex) New York

    Returns:
      region_code (str): 지역의 코드 ex) US_NY
    '''
    index = pd.read_csv('./data/us_index.csv')
    
    if region_name == 'US':
        return region_name

    region_code = index[(index['subregion1_name'] == region_name) &
                      (index['aggregation_level'] == 1)]['key'].values[0]
    return region_code.replace("_", "-")
#end def

# code > name
def convRegionCode2Name(region_codes):
    ''' 지역 코드를 이름으로 변환하는 함수
    Args:
      region_code (str): 지역의 코드

    Returns:
      region_name (str): 지역의 이름
    '''
    index = pd.read_csv('./data/us_index.csv')
    region_code = region_codes.replace("-", "_")
    
    if region_code == 'US':
        return region_code

    country_code, subregion_code = region_code.split('_')
    region_name = index[(index['country_code'] == country_code) &
                      (index['subregion1_code'] == subregion_code)]['subregion1_name'].values[0]
    return region_name
#end def
