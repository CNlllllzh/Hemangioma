import os
import shutil
from sklearn.model_selection import train_test_split

# 图像和标签的原始路径
original_img_dir = r"E:\ML Data\Hemangioma\train_dataset\img"
original_mask_dir = r"E:\ML Data\Hemangioma\train_dataset\mask"

# 划分后的数据集存储路径
base_dir = r"E:\ML Data\Hemangioma\train_dataset"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 创建train, val, test文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有图像文件的路径
img_files = [f for f in os.listdir(original_img_dir) if os.path.isfile(os.path.join(original_img_dir, f))]
img_paths = [os.path.join(original_img_dir, f) for f in img_files]

# 使用图像文件名来获取对应的标签文件路径
mask_paths = [os.path.join(original_mask_dir, os.path.splitext(f)[0] + '.png') for f in img_files]

# 将图像和标签路径合并为一个列表，以便一起进行划分
img_mask_pairs = list(zip(img_paths, mask_paths))

# 划分训练集和临时集
train_pairs, temp_pairs = train_test_split(img_mask_pairs, test_size=0.4, random_state=42)

# 进一步将临时集划分为验证集和测试集
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

# 定义一个函数来复制图像和标签文件到目标文件夹
def copy_files(pairs, target_dir):
    for img_path, mask_path in pairs:
        shutil.copy(img_path, os.path.join(target_dir, 'img'))
        shutil.copy(mask_path, os.path.join(target_dir, 'mask'))

# 创建子文件夹img和mask
os.makedirs(os.path.join(train_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'mask'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'mask'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'mask'), exist_ok=True)

# 复制文件到对应的文件夹
copy_files(train_pairs, train_dir)
copy_files(val_pairs, val_dir)
copy_files(test_pairs, test_dir)

# 打印每个集合的大小
print(f"训练集大小: {len(train_pairs)}")
print(f"验证集大小: {len(val_pairs)}")
print(f"测试集大小: {len(test_pairs)}")
