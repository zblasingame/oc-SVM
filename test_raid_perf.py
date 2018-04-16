import utils.datasets as ds
import models.svm as svm
import json
import numpy as np
import os
import plotly.plotly as py
import plotly.graph_objs as go


# Data format
data_format = 'ts'
DATA_FORMAT = 'Time Series'

model = svm.SVM(debug=True)

loc_name = 'raid/oc-svm/{}'.format(data_format)
loc_str = 'results/{}/'.format(loc_name)
loc_str += '{}.json'
path = 'results/{}'.format(loc_name)
if not os.path.exists(path):
    os.mkdir(path)

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

for exploit in exploits:
    data = []

    for i in range(5):
        X_train, _ = ds.load_data(
            (
                './data/raid/{}/{}/subset_{}/train_set.csv'
            ).format(data_format, exploit, i)
        )

        model.train(X_train)

        for j in range(5):
            X_test, Y_test = ds.load_data(
                (
                    './data/raid/{}/{}/subset_{}/test_set.csv'
                ).format(data_format, exploit, i)
            )

            data.append(model.test(X_test, Y_test))

    with open(loc_str.format(exploit), 'w') as f:
        json.dump(data, f, indent=2)

# Plot stuff

net_data = {}

for ex in exploits:
    with open('{}/{}.json'.format(path, ex)) as f:
        net_data[ex] = json.load(f)

# plot data
accs = [[entry['accuracy'] for entry in net_data[ex]] for ex in exploits]

boxes = [go.Box(
    y=accs[i],
    name=exploits[i],
    boxmean='sd'
) for i in range(len(exploits))]

layout = go.Layout(
    title='OC-SVM: Accuracy per Exploit with {} Data'.format(DATA_FORMAT),
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=boxes, layout=layout)
py.plot(fig, filename='raid/data-format-perfs/svm/net-results-{}'.format(
    data_format
))
