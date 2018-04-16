import json
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

path = './results/svm_test/net_results/'
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
    title='OC-SVM: Accuracy per Exploit',
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=boxes, layout=layout)
py.plot(fig, filename='raid/model-tests/svm/svm-test')
