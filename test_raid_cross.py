import utils.datasets as ds
import models.svm as svm
import csv
import numpy as np


model = svm.SVM(debug=True)

exploits = [
    ('nginx_keyleak', 'nginx_rootdir'),
    ('nginx_rootdir', 'nginx_keyleak'),
    ('orzhttpd_rootdir', 'orzhttpd_restore'),
    ('orzhttpd_restore', 'orzhttpd_rootdir')
]


data = []

for train_ex, test_ex in exploits:
    for i in range(5):
        X_train, _ = ds.load_data(
            (
                './data/raid/ts/{}/subset_{}/train_set.csv'
            ).format(train_ex, i)
        )

        model.train(X_train)

        for j in range(5):
            X_test, Y_test = ds.load_data(
                (
                    './data/raid/ts/{}/subset_{}/test_set.csv'
                ).format(test_ex, j)
            )

            d = model.test(X_test, Y_test)
            tmp = ['svm', train_ex, test_ex]
            tmp.append(d['accuracy'])
            tmp.extend(np.array(d['confusion_matrix']).flatten().tolist())
            data.append(tmp)

data = np.array(data)

with open('./results/raid/cross/data.csv', 'w') as f:
    writer = csv.writer(f)

    for row in data:
        writer.writerow(row)
