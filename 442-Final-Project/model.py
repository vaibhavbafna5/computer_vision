import numpy as np
import requests
import pprint
import datetime as dt
from pymongo import MongoClient
import torch
from torch.autograd import Variable
from skimage import transform

from PIL import Image
from io import BytesIO
import math

client = MongoClient("mongodb+srv://sunset-data-manager-admin:sunset442@cluster0-stvht.mongodb.net/test?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['ImageMetaData']
image_collection = db['Images_2']
docs = list(image_collection.find())
# print(len(docs))
# exit(1)
input = []
label = []
for doc in docs:
    if 'minutes_until_sunset' in doc and abs(doc['minutes_until_sunset']) < 180:
        try:
            response = requests.get(doc['image_url'])
            # print(doc['image_url'])
            image = Image.open(BytesIO(response.content))
            image.seek(0)
            low_res_img = image.resize((64, 64))
            arr = np.array(low_res_img)
            arr = arr.flatten()
            input.append(arr)
            label.append([doc['minutes_until_sunset']])
        except:
            pass

input = np.array(input).astype(np.float32)
label = np.array(label).astype(np.float32)
train_end = 4*(len(input)//5)

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 12288
outputDim = 1
learningRate = 1e-9
epochs = 10

model = linearRegression(inputDim, outputDim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

print('\nTRAINING')
print('--------------\n')
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    # if torch.cuda.is_available():
    #     inputs = Variable(torch.from_numpy(x_train).cuda())
    #     labels = Variable(torch.from_numpy(y_train).cuda())
    # else:
    inputs = Variable(torch.from_numpy(input[:train_end]))  # input is flattened array
    labels = Variable(torch.from_numpy(label[:train_end]))  # output is time difference
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
    print("AVERAGE DEVIATION FROM ACTUAL minutes_until_sunset: ", math.sqrt(loss.item()))

print('\nTESTING')
print('--------------\n')
with torch.no_grad():
    # if torch.cuda.is_available():
    #     predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu()
    # else:
    predicted = model(Variable(torch.from_numpy(input[train_end:])))
    labels = Variable(torch.from_numpy(label[train_end:]))
    loss = criterion(predicted, labels)
    print("TOTAL LOSS FOR TEST SET: ", loss.item())
    print("AVERAGE DEVIATION FROM ACTUAL minutes_until_sunset: ", math.sqrt(loss.item()))
