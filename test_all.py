import utils.datasets as ds
import models.svm
import json
import numpy as np
import os


hyperparameters = dict(
    num_features=12,
    debug=True
)

model = models.svm.SVM(**hyperparameters)

loc_name = 'svm-ts/net_results'
loc_str = 'results/{}/'.format(loc_name)
loc_str += '{}.json'
loc = 'results/{}'.format(loc_name)
if not os.path.exists(loc):
    os.mkdir(loc)

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

for exploit in exploits:
    data = []

    for i in range(5):
        X_train, _ = ds.load_data(
            (
                './data/raid/sv/{}/subset_{}/train_set.csv'
            ).format(exploit, i)
        )

        model.train(X_train)

        for j in range(5):
            X_test, Y_test = ds.load_data(
                (
                    './data/raid/sv/{}/subset_{}/test_set.csv'
                ).format(exploit, j)
            )

            data.append(model.test(X_test, Y_test))

    with open(loc_str.format(exploit), 'w') as f:
        json.dump(data, f, indent=2)
