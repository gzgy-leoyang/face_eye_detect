import cv2


# 识别电脑摄像头并打开
cap = cv2.VideoCapture(0)
# 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
# Haar特征分类器就是一个XML文件，该文件中会描述人体各个部位的Haar特征值。包括人脸、眼睛、嘴唇
face_detect = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier(r'./haarcascade_eye.xml')

while True:
    # 读取视频片段
    flag, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not flag:
        break
    # 灰度处理
    # cv2.cvtColor(src, code[, dst[, dstCn]])
    # src:输入图像
    # code:色彩空间转换代码
    # dst:输出图像，可选
    # dstCn:输出图像中的频道数，可选
    # 返回值：输出图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detectMultiScale函数 ,输入一张灰度图，多尺度检测,返回所有检测目标的rectangle的数组
    # 检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
    # image输入图像
    # objects:人脸目标序列
    # scaleFactor:图像尺寸减小的比例
    # minNeighbors: 表示每一个目标至少要被检测到n次才算是真的目标
    # minSize:最小尺寸
    # minSize:最大尺寸
    face_vector = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # 绘制矩形和圆形检测人脸
    for x, y, w, h in face_vector:
        img = cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 128, 255], thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes_vector = eye_detect.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2) )
        for (ex, ey, ew, eh) in eyes_vector:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    # 显示图片
    cv2.imshow('face', frame)
    # 设置退出键q 展示频率
    if ord('q') == cv2.waitKey(30):
        break

# 释放资源
cv2.destroyAllWindows()
cap.release()