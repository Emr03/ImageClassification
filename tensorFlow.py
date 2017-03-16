import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join


#folderPath = '/home/snedogisawesome/Ima'

#files = [f for f in listdir(folderPath) if isFile(join(folderPath, f))]
#imagePath = '/home/snedogisawesome/ImageClassification/images_test/img_CV2_99.jpg'
folderPath = '/home/snedogisawesome/ImageClassification/images_test/'

Images = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]

modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _=tf.import_graph_def(graph_def, name='')
k = 0
for imagePath in range(0, 5):
    imagePath = '/home/snedogisawesome/ImageClassification/images_test/img_CV2_'+str(imagePath)+'.jpg'
    pred_list = []
    def predict():
        answer = None

        if not tf.gfile.Exists(imagePath):
            tf.logging.fatal('File does not exist %s', imagePath)
            return answer

        image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
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
            pred_list.append(answer)

    if __name__ == '__main__':
        predict()
    print('Iteration '+str(k)+' complete')
    k += 1
pred=np.array(pred_list)
np.save('pred', pred)
