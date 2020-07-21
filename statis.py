# %%
import json

# %%
json_path = "./data/amap_traffic_annotations_train.json"

'''
"status": 0 畅通，1 缓行，2 拥堵，-1 未知（测试集默认状态为-1）

annotations[0]
{'id': '000001',
 'key_frame': '3.jpg',
 'status': 0,
 'frames': [{'frame_name': '1.jpg', 'gps_time': 1552212488},
  {'frame_name': '2.jpg', 'gps_time': 1552212493},
  {'frame_name': '3.jpg', 'gps_time': 1552212498},
  {'frame_name': '4.jpg', 'gps_time': 1552212503}]}
'''

# %%
with open(json_path, "r", encoding="utf-8") as f:
    json_dict = json.load(f)
    annotations = json_dict["annotations"]  
    cnt = {3:0, 4:0, 5:0}
    for anno in annotations:
        cnt[len(anno['frames'])] += 1

    # cnt : {3: 3, 4: 602, 5: 895}
# %%
