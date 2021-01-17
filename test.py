from torch import nn
import torch
# from online_triplet_loss.losses import *
# from loss import *

# model = nn.Embedding(10, 10)
# labels = torch.randint(high=10, size=(5,)) # our five labels
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择在cpu或cuda运行
# for i in range(100):
#     embeddings = model(labels)
#     # print('Labels:', labels)
#     # print('Embeddings:', embeddings)
#     loss = batch_hard_triplet_loss(labels, embeddings, margin=100, device=device)
#     print('Loss:', loss)
#     loss.backward()

# from loss import batch_hard_triplet_loss
#
# model = nn.Embedding(10, 10)
# labels = torch.randint(high=10, size=(5,)) # our five labels
# labels = torch.cat((labels, labels), dim=0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择在cpu或cuda运行
# embeddings = model(labels)
# print('Labels:', labels)
# # print('Embeddings:', embeddings)
# loss = batch_hard_triplet_loss(5, labels, embeddings, margin=100, device=device)
# print('Loss:', loss)
# loss.backward()

# index_dict = dict([(i, 0) for i in range(0, 10)])
# for k, v in index_dict.items():
#     print(k, v)

from dataset import TripletDataset
dataset = TripletDataset(num_sub_dataset=2)
print(len(dataset))

# import numpy as np
# np_data = np.random.randint(0, 100, size=(3, 4, 3))
# torch_data = torch.from_numpy(np_data)
# print(torch_data)
# torch_data = torch_data.view((-1,torch_data.size(-1)))
# print(torch_data)

# import numpy as np
# np_data_anc = np.asarray([0.43, 0.39, 0.54, 0.21, 0.50, 0.00, 0.53, 0.27])
# np_data_pos = np.asarray([0.64, 0.62, 0.49, 0.15, 0.50, 0.00, 0.53, 0.22])
# np_data_neg = np.asarray([0.85, 0.56, 0.33, 0.38, 1.00, 0.00, 0.55, 0.25])
# print(np.sqrt(np.sum((np_data_anc - np_data_pos)**2)))
# print(np.sqrt(np.sum((np_data_anc - np_data_neg)**2)))

# import numpy as np
#
# X = np.asarray([[0, 3], [1, 5], [2, 3], [3, 4]])
# y = np.asarray([6, 1, 4, 4])
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)  # doctest: +ELLIPSIS
# print(neigh.predict([[0.9, 0.8]]))
# print(neigh.predict_proba([[0.9, 0.8]]))

# import numpy as np
#
# x = np.zeros((0,))
# y = np.asarray([2, 3, 4])
# y.astype(int)
# print(np.append(x, y))

# import numpy as np
#
# c = np.array([[11, 2, 8, 4], [4, 52, 6, 17], [2, 8, 9, 100]])
#
# print(np.argmax(c, axis=0))  # 按每列求出最大值的索引
# print(np.argmax(c, axis=1))  # 按每行求出最大值的索引

# import numpy as np
# np_data = np.asarray([2, 3, 1])
# torch_data = torch.from_numpy(np_data)
# print(torch_data)
# torch_data = torch.cat(tuple([torch_data for i in range(3)]), dim=0)
# print(torch_data)