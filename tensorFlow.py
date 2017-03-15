import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join


folderPath = '/home/snedogisawesome/Ima'

files = [f for f in listdir(folderPath) if isFile(join(folderPath, f))]
imagePath = '/tmp/imagenet/flower.jpg'
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _=tf.import_graph_def(graph_def, name='')


def predict():
    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    answer = None
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':
    predict()