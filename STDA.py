import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

def stda(X, y, num_components):
    # X: テンソルデータ (サンプル数 x 次元1 x 次元2 x ... x 次元n)
    # y: クラスラベル (サンプル数,)
    # num_components: 低次元の特徴空間の次元数

    # 各クラスのインデックスを取得
    classes = np.unique(y)
    num_classes = len(classes)

    # 各クラスごとにテンソルのスライスを計算
    slices = []
    for c in classes:
        X_c = X[y == c]
        slice_c = np.mean(X_c, axis=0)
        slices.append(slice_c)

    # 各クラスのスライスから共分散行列を計算
    covariance_matrices = [np.cov(slice_c.reshape(slice_c.shape[0], -1).T) for slice_c in slices]

    # STDAの基底ベクトルを計算
    total_covariance = np.mean(covariance_matrices, axis=0)
    within_covariance = np.mean(covariance_matrices, axis=0)
    between_covariance = total_covariance - within_covariance

    # between_covariance と within_covariance を足し合わせた行列を作成
    covariance_matrix = between_covariance + within_covariance

    # eig()関数に covariance_matrix を渡して計算
    _, U = la.eig(covariance_matrix)

    components = U[:, :num_components]

    # データを低次元の特徴空間に射影
    projected_data = np.array([np.dot(X[i].reshape(X[i].shape[0], -1), components) for i in range(X.shape[0])])

    return projected_data


# テンソルデータとクラスラベルを用意
# ここではダミーデータを使用します。実際のデータに置き換えてください。

"""
X = np.random.rand(5, 5, 5)  # サンプル数 x 次元1 x 次元2 x ... x 次元nの形式のテンソルデータ
y = np.random.randint(0, 3, 5)  # クラスラベル (0または1)
    
"""


df=pd.read_csv('./K_log/2014_drop.csv')

months=[30,31,30,31,31,30,25,30,31,31,28,28]#2014の含まれている日数,4月から順番に最後が3月、10月は9日から14日が欠けている,4:0,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,1:9,2:10,3:11
count=4
y=[]
for i in months:
    for j in range(i):
        y.append(count)
    count=count+1
    if(count>12):
        count=1

#print(y)    

# 1. prepare data
dataset = df
X = dataset

X = X[['AirIn','AirOut', 'CPU', 'Water']]

#feat_names = ['AirIn','AirOut', 'CPU', 'Water']

# replace to a shorter name
#feat_names[3] = 'od280/od315'


# normalization
X = preprocessing.scale(X)

X = X.reshape(-1, 864, 4)

y=np.array(y)

#print(y)

# STDAを実行して低次元の特徴空間に射影
num_components = 2  # 低次元の特徴空間の次元数
projected_data = stda(X, y, num_components)
print(projected_data.shape)




# データを平均化して(864, 2)のデータに変換
averaged_data = np.mean(projected_data, axis=0)

# (864, 2)のデータを(36, 24, 2)の形に変形
reshaped_data = averaged_data.reshape(36, 24, 2)

# 2画面に分けて2つのチャンネルのヒートマップを表示
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# チャンネル0のヒートマップを表示
axes[0].imshow(reshaped_data[:, :, 0].T, cmap='viridis', aspect='auto')
axes[0].set_title('Channel 0')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_xticks([])
axes[0].set_yticks([])

# チャンネル1のヒートマップを表示
axes[1].imshow(reshaped_data[:, :, 1].T, cmap='viridis', aspect='auto')
axes[1].set_title('Channel 1')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_xticks([])
axes[1].set_yticks([])

# グラフを表示
plt.tight_layout()
plt.show()


"""
# プロットしてヒートマップを作成
plt.figure(figsize=(8, 6))
plt.scatter(projected_data[y == 0, 0], projected_data[y == 0, 1], c='red', label='Class 0')
plt.scatter(projected_data[y == 1, 0], projected_data[y == 1, 1], c='blue', label='Class 1')
plt.scatter(projected_data[y == 2, 0], projected_data[y == 2, 1], c='green', label='Class 2')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.title('STDA Projection')
plt.colorbar()
plt.show()    
"""

"""
#2画面で[:,:,0],[:,:,1]をヒートマップでプロット
channel_0 = projected_data[:, :, 0]
plt.subplot(1, 2, 1)
plt.imshow(channel_0, cmap='viridis', aspect='auto')
plt.title('Channel 0')
plt.colorbar()

# チャンネル1のヒートマップ
channel_1 = projected_data[:, :, 1]
plt.subplot(1, 2, 2)
plt.imshow(channel_1, cmap='viridis', aspect='auto')
plt.title('Channel 1')
plt.colorbar()

# グラフを表示
plt.tight_layout()
plt.show()
"""   

"""
# 12個のグループにデータを分割
num_groups = 13
#group_indices = np.linspace(0, data_points, num_groups+1, dtype=int)

# グループごとに異なる色を割り当てる
colors = plt.cm.jet(np.linspace(0, 1, num_groups))
# 散布図をプロット
for i in y:
    plt.scatter(projected_data[y == i, 0], projected_data[y == i, 1], color=colors[i])

# 凡例を表示
plt.legend()

# グラフを表示
plt.show()    
    
"""
  


