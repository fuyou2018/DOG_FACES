%matplotlib inline




input_dir='./data/face_recog/test_faces'
index=1

output = cnnLayer()
predict = tf.argmax(output, 1)

#先加载 meta graph并恢复权重变量
saver = tf.train.import_meta_graph('./train_face_model/train_faces.model.meta')
sess = tf.Session()

saver.restore(sess, tf.train.latest_checkpoint('./train_face_model/'))
#saver.restore(sess,tf.train.latest_checkpoint('./my_test_model/'))

def is_my_face(image):
sess.run(tf.global_variables_initializer())
res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
if res[0] == 1:
   return True
else:
   return False

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

#cam = cv2.VideoCapture(0)

#while True:
#_, img = cam.read()
for (path, dirnames, filenames) in os.walk(input_dir):
  for filename in filenames:
     if filename.endswith('.jpg'):
     print('Being processed picture %s' % index)
     index+=1
     img_path = path+'/'+filename
# 从文件读取图片
img = cv2.imread(img_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets = detector(gray_image, 1)
if not len(dets):
print('Can`t get face.')
cv2.imshow('img', img)
key = cv2.waitKey(30) & 0xff
if key == 27:
sys.exit(0)
for i, d in enumerate(dets):
x1 = d.top() if d.top() > 0 else 0
y1 = d.bottom() if d.bottom() > 0 else 0
x2 = d.left() if d.left() > 0 else 0
y2 = d.right() if d.right() > 0 else 0
face = img[x1:y1,x2:y2]
# 调整图片的尺寸
face = cv2.resize(face, (size,size))
print('Is this my face? %s' % is_my_face(face))
cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
cv2.imshow('image',img)
key = cv2.waitKey(30) & 0xff
if key == 27:
sys.exit(0)

sess.close()
