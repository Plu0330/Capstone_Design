import random
import os
import numpy as np
import warnings
import json
import sys
import pickle
import time

warnings.filterwarnings("ignore")
base_path = '/mnt/c/Users/hyejuchoi/Desktop/RoadtenceGenerator - 2808'
file_store_path ='/mnt/d/chj/Roadtence_v2_2808'

initialize = 0

now = time.gmtime(time.time())
print(f'{now.tm_year}.{now.tm_mon}.{now.tm_mday}')
print(f'{now.tm_hour}:{now.tm_min}:{now.tm_sec}')


class AngleCal: 
    def __angle_between(self, p1, p2):  #두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        return res

    def getAngle3P(self, p1, p2, p3, direction="CW"): #세점 사이의 각도 1->2->3
        pt1 = (p1[0] - p2[0], p1[1] - p2[1])
        pt2 = (p3[0] - p2[0], p3[1] - p2[1])
        res = self.__angle_between(pt1, pt2)
        res = (res + 360) % 360
        if direction == "CCW":    #반시계방향
            res = (360 - res) % 360

        angle_sector, sec, sec_idx = self.getSector(res)
        long1, long2 = self.getLongDisAngle(angle_sector, res)
        
        return res, sec, sec_idx, long1, long2

    def getSector(self, angle: float):  #시작 방향에 따라 달라진다. 현재는 북쪽 방향을 기준으로 시계방향 360도임

        sectors = [i*45 for i in range(0,9)]
        
        sec_list = [f'{i+1}구역 ({sectors[i]}°~ {sectors[i+1]}°)' for i in range(0,8)]     
        sector_pairs = [(sectors[i],sectors[i+1]) for i in range(0,8)]
        
        sec = sec_list[int(angle//45)]
        
        sector_p = sector_pairs[int(angle//45)]
        
        sector_idx = sec_list.index(sec)
        
        return sector_p, sec, sector_idx
        
    def getLongDisAngle(self, sector: set, short_ang):
        long_ang1 = short_ang - 11.25
        long_ang2 = short_ang + 11.25

        if long_ang1<sector[0]:
            long_ang1=sector[0]
        if sector[1]<long_ang2:
            long_ang2=sector[1]
        
        return long_ang1, long_ang2



############ 근거리 설정 후 원거리 범위 각 구하기 

class RelatedRegion:
    
    def __init__(self, region: str, limit: int):
        self.region_name = region
        self.distance_limit = limit
        self.short_d_n = []
        self.short_d = []
        self.short_x = []
        self.short_y = []
        self.temp_d_n, self.temp_d, self.temp_x, self.temp_y, self.temp_address = [],[],[],[],[]
        self.region_set_list = [[],[],[],[],[],[],[],[]]
        print(os.getcwd())
        self.x_coord_file = f"./region_x.txt"#경도
        self.y_coord_file = f"./region_y.txt"#위도
        self.n_coord_file = f"./region_name.txt"#region name
        self.address_file = f"./region_coord.txt"#coord

        f1 = open(self.x_coord_file,"r",encoding='UTF8')
        f2 = open(self.y_coord_file,"r",encoding='UTF8')
        f3 = open(self.n_coord_file,"r", encoding='UTF8')
        f4 = open(self.address_file,"r",encoding='UTF8')

        self.x_coord = f1.read() #lon
        self.y_coord = f2.read() #lat
        self.n_coord_list = f3.readlines()
        self.address_list = f4.readlines()

        self.x_coord_list = self.x_coord.replace('\n',' ').split(' ') 
        self.y_coord_list = self.y_coord.replace('\n',' ').split(' ')


        self.x_coord_list = [float(i) for i in self.x_coord_list]
        self.x_coord_list = list(filter(None, self.x_coord_list))
        self.y_coord_list = [float(i) for i in self.y_coord_list]
        self.y_coord_list = list(filter(None, self.y_coord_list))

        self.n_coord_list = [i.replace('\n','') for i in self.n_coord_list]
        self.n_coord_list = list(filter(None, self.n_coord_list))

        self.address_list = [i.replace('\n','') for i in self.address_list]
        self.address_list = list(filter(None, self.address_list))
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{region}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    def coord_finder(self, any_region: str):
        region_name = any_region
        region_idx = self.n_coord_list.index(region_name)
        region_x = self.x_coord_list[region_idx]
        region_y = self.y_coord_list[region_idx]
        return region_x, region_y, region_name

    def start_point(self):
        return self.coord_finder(self.region_name)
    
    def region_find(self):
        return self.short_region()
    
    @staticmethod
    def search_distance(address: str ,x: float, y:float):
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{address}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        search_info_f = f"./temp.txt"
        os.popen(f'bash search_distance.sh "{address}" "{x}" "{y}">temp.txt').read()
        print(f'bash search_distance.sh "{address}" "{x}" "{y}">temp.txt')
        f=open(search_info_f,"r", encoding='UTF8')
        f_list = f.read()
        print(f_list)
        f_list = f_list.split(',')
        dist = f_list[-2]
        dist = dist.replace('}]','').split(':')
        dist = float(dist[1])
        
        return dist
    
    def short_region(self):######### distance 데이터 구하기 start
        unit = 2000
        start_x, start_y, name = self.start_point()      
        
        for n,i in enumerate(self.n_coord_list):
         
            distance = self.search_distance(self.address_list[n],start_x,start_y)

            x = self.x_coord_list[n] # x좌표가 저장된 열
            y = self.y_coord_list[n] # y좌표가 저장된 열
            
            if distance<=self.distance_limit:
                self.short_d_n.append(i)
                self.short_d.append(float(distance))
                self.short_x.append(x)
                self.short_y.append(y)
            else:
                self.temp_d_n.append(i)
                self.temp_d.append(float(distance))
                self.temp_x.append(x)
                self.temp_y.append(y)
                self.temp_address.append(self.address_list[n])
                
        for i in range(len(self.short_d_n)):
            center,sector,sector_idx, short_point, long_point = self.angle_range(name, start_x, start_y, self.short_d_n[i], self.short_x[i], self.short_y[i])
            if short_point!=long_point and short_point!=center and short_point!=long_point:
                self.region_set_list[sector_idx].append((center,short_point,long_point))
# ------------------------------------ 1차 탐색 끝
                print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!\n BEFORE expansion: {self.region_set_list}\n !!!!!!!!!!!!!!!!!!!!!!!!!! ')

# ------------------------------------ 2차 탐색 시작
        for n,i in enumerate(self.region_set_list): # region_set_list 중
            for expand_num in range(1,6): # 10 번 돌면서
                if len(i)==0: # len(region_set_list[n]) 이 비어있는 경우
                    for c_n,j in enumerate(self.temp_d_n): # n_coord_list 에서
                        center,sector,sector_idx, short_point, long_point = self.angle_range(name, start_x, start_y, j, self.temp_x[c_n], self.temp_y[c_n]) # angle 에 따른 sector 정보를 얻어내
                        if sector_idx==n: #sector 가 region_set_list 의 n 번째라면
                            distance = self.search_distance(self.temp_address[c_n],start_x,start_y) # distance 를 검색한다.
                            if distance < self.distance_limit + unit*expand_num:                 # distance 가 limit + unit*expand_num (1000 단위 증가) 보다 작을 경우
                                if short_point!=long_point and short_point!=center and short_point!=long_point: # short_point, long_point, center 가 각각 다른 region 인지 확인하고 append
                                    self.region_set_list[sector_idx].append((center,short_point,long_point))
                else:
                    continue
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!\n AFTER expansion: {self.region_set_list}\n !!!!!!!!!!!!!!!!!!!!!!!!!! ')
        ######### distance 데이터 구하기 end

        return self.region_set_list
    
    ############ 원거리 범위각 내 가장 멀고, 근거리보다 먼 region    
    def angle_range(self, name1, start1,start2, name2, x, y):
    ############ 원거리 범위각 내 포함 되는 (center, 특정 근거리) 기준 region 찾기
        angle = AngleCal()
        
        res, sect, sect_idx, l_d1, l_d2 = angle.getAngle3P((start1, start2 + 0.2), (start1, start2), (x,y))
        
        cnt = 0
        long_d = 0
        long_d_n = name2 # 근거리 region 으로 초기화
        
        for i in range(len(self.short_d_n)): # 행 개수만큼 순회

            name = self.short_d_n[i] # 이름이 저장된 열
            if len(name)==0:
                continue
            x = self.short_x[i] # x좌표가 저장된 열
            y = self.short_y[i] # y좌표가 저장된 열

            temp_res, sect,_, _, _ = angle.getAngle3P((start1, start2 + 0.2), (start1, start2), (x,y))

            if l_d1<temp_res:
                cnt += 1
            if temp_res<l_d2:
                cnt += 1

            if cnt==2:
                if self.short_d[i]>long_d:
                    long_d_n = self.short_d_n[i]
                    long_d = self.short_d[i]

            cnt = initialize

        long_d_x, long_d_y, name = self.coord_finder(long_d_n)
        
        return name1, sect, sect_idx, name2, long_d_n

center_region = sys.argv[1]
region_set = RelatedRegion(center_region, 5500)
print(region_set.region_find())

combination = [[],[],[],[],[],[],[],[]]

for n,i in enumerate(getattr(region_set,'region_set_list')):
    combination[n] = [j for j in range(len(i)) ]

sample_n = 12
roadtence_make_num=5000

combination_sample = []
print(combination)

for i in range(len(combination)):
    if len(combination[i]) >= sample_n:
        combination_sample.append(random.sample(combination[i], sample_n))
    elif len(combination[i]) < sample_n and len(combination[i]) != 0:
        combination_sample.append(random.sample(combination[i], len(combination[i])))
    elif len(combination[i]) == 0:
        continue

new_array = np.array(np.meshgrid(*combination_sample)).T.reshape(-1,len(combination_sample))

region_list = getattr(region_set,'region_set_list')
f = open(f"{file_store_path}/{center_region}_sample{sample_n}_{roadtence_make_num}.txt", "w")

temp_list = []
roadtence_idx = 0


valid_region = [v for v in region_list if len(v)!=0]

new_array_choice=[]
for _ in range(roadtence_make_num):
    x=random.choice(new_array)
    new_array_choice.append(x)

for array_n, i in enumerate(new_array_choice): # index 조합 list
    direction = random.choice(range(len(i)))
    #for k in range(len(i)):        # 8 분할 방향
    for n,j in enumerate(i): # index 조합 중 진행방향에 대해 원거리 삽입 n: 섹터, j:조합 index
        temp_list.append(valid_region[n][j][1])
        if n==direction:
            temp_list.append(valid_region[n][j][2])
    if direction>0:
        temp_list = temp_list[direction:] + temp_list[:direction]
    temp_list.insert(0,center_region)
    temp_roadtence = ' '.join(temp_list)
    #f.write(f'{roadtence_idx} : {temp_roadtence}\n')
    f.write(f'{temp_roadtence}\n')
    temp_list = []
    roadtence_idx += 1

f.close()

now = time.gmtime(time.time())
print(f'{now.tm_year}.{now.tm_mon}.{now.tm_mday}')
print(f'{now.tm_hour}:{now.tm_min}:{now.tm_sec}')