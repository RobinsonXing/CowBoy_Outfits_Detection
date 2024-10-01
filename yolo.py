import json
import os
import yaml
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image

# 读取数据集
json_file_path = './dataset/train.json'
yolo_anno_path = './training/yolo_anno/'

data = json.load(open(json_file_path, 'r'))

# 创建 YOLO 注释文件夹
if not os.path.exists(yolo_anno_path):
    os.makedirs(yolo_anno_path)

# 创建类别 ID 映射
cate_id_map = {}
num = 0
for cate in data['categories']:
    cate_id_map[cate['id']] = num
    num += 1

# 坐标转换函数：COCO to YOLO
def cc2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    return x * dw, y * dh, w * dw, h * dh

# 创建 CSV 文件，准备图像与注释
f = open('./training/train.csv', 'w')
f.write('id,file_name\n')

for i in tqdm(range(len(data['images']))):
    filename = data['images'][i]['file_name']
    img_width = data['images'][i]['width']
    img_height = data['images'][i]['height']
    img_id = data['images'][i]['id']
    yolo_txt_name = filename.split('.')[0] + '.txt'  # 去除扩展名

    f.write('{},{}\n'.format(img_id, filename))
    yolo_txt_file = open(os.path.join(yolo_anno_path, yolo_txt_name), 'w')

    for anno in data['annotations']:
        if anno['image_id'] == img_id:
            yolo_bbox = cc2yolo_bbox(img_width, img_height, anno['bbox'])
            yolo_txt_file.write('{} {} {} {} {}\n'.format(cate_id_map[anno['category_id']], *yolo_bbox))
    yolo_txt_file.close()
f.close()

# 分割训练集与验证集
train = pd.read_csv('./training/train.csv')
train_df, valid_df = train_test_split(train, test_size=0.10, random_state=233)

train_df['split'] = 'train'
valid_df['split'] = 'valid'
df = pd.concat([train_df, valid_df]).reset_index(drop=True)

# 创建必要的目录
os.makedirs('./training/cowboy/images/train', exist_ok=True)
os.makedirs('./training/cowboy/images/valid', exist_ok=True)
os.makedirs('./training/cowboy/labels/train', exist_ok=True)
os.makedirs('./training/cowboy/labels/valid', exist_ok=True)

# 复制图像和标签
for i in tqdm(range(len(df))):
    row = df.loc[i]
    name = row.file_name.split('.')[0]
    if row.split == 'train':
        copyfile(f'./dataset/images/{name}.jpg', f'./training/cowboy/images/train/{name}.jpg')
        copyfile(f'./training/yolo_anno/{name}.txt', f'./training/cowboy/labels/train/{name}.txt')
    else:
        copyfile(f'./dataset/images/{name}.jpg', f'./training/cowboy/images/valid/{name}.jpg')
        copyfile(f'./training/yolo_anno/{name}.txt', f'./training/cowboy/labels/valid/{name}.txt')

# 生成 YOLO 的数据集配置文件
data_yaml = dict(
    train='./training/cowboy/images/train/',
    val='./training/cowboy/images/valid/',
    nc=5,
    names=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']
)

with open('./training/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)

# 训练模型
BATCH_SIZE = 32
EPOCHS = 5
MODEL = 'yolov5m.pt'
name = f'{MODEL}_BS_{BATCH_SIZE}_EP_{EPOCHS}'

os.system(f'python ./training/yolov5/train.py --batch {BATCH_SIZE} '
          f'--epochs {EPOCHS} --data /kaggle/training/yolov5/data/data.yaml '
          f'--weights {MODEL} --save-period 1 --project ./working/kaggle-cowboy --name {name} --cache-images')

# 推理和结果转换
PRED_PATH = './training/yolov5/runs/detect/exp/labels/'
valid_df = pd.read_csv('./dataset/cowboyoutfits/valid.csv')

def yolo2cc_bbox(img_width, img_height, bbox):
    x = (bbox[0] - bbox[2] * 0.5) * img_width
    y = (bbox[1] - bbox[3] * 0.5) * img_height
    w = bbox[2] * img_width
    h = bbox[3] * img_height
    return x, y, w, h

# 生成提交文件
def make_submission(df, PRED_PATH, IMAGE_PATH):
    output = []
    for i in tqdm(range(len(df))):
        row = df.loc[i]
        image_id = row['id']
        file_name = row['file_name'].split('.')[0]
        if f'{file_name}.txt' in os.listdir(PRED_PATH):
            img = Image.open(f'{IMAGE_PATH}/{file_name}.jpg')
            width, height = img.size
            with open(f'{PRED_PATH}/{file_name}.txt', 'r') as file:
                for line in file:
                    preds = list(map(float, line.strip().split(' ')))
                    cc_bbox = yolo2cc_bbox(width, height, preds[1:-1])
                    result = {'image_id': image_id, 'category_id': int(preds[0]), 'bbox': cc_bbox, 'score': preds[-1]}
                    output.append(result)
    return output

sub_data = make_submission(valid_df, PRED_PATH, './dataset/cowboyoutfits/images')
op_pd = pd.DataFrame(sub_data)
op_pd.to_json('./working/answer.json', orient='records')

# 压缩提交文件
import zipfile
zf = zipfile.ZipFile('./working/sample_answer.zip', 'w')
zf.write('./working/answer.json', 'answer.json')
zf.close()