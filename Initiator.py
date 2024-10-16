import cv2
from deepface import DeepFace
import random

emo_list = ['高兴的', '中立的', '伤心的', '惊讶的', '愤怒的', '厌恶的', '恐惧的']
emo_dict = {'happy': '高兴的', 'neutral': '中立的', 'sad': '伤心的', 'surprise': '惊讶的',
            'angry': '愤怒的', 'disgust': '厌恶的', 'fear': '恐惧的'}

global cn_emo1, cn_emo2


def ImgLocal():
    global cn_emo1
    image_path = input("请输入图片路径(路径不需用引号包裹)：")
    img = cv2.imread(image_path)
    res = DeepFace.analyze(image_path, actions=['emotion'])
    cons = res[0]["face_confidence"]
    cn_emo1 = emo_dict[res[0]["dominant_emotion"]]
    print(f"图片中的人物表情有{cons}的概率是{cn_emo1}。")
    cv2.imshow("Emotion Picture", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cn_emo1


def VidCap():
    global cn_emo2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧，请检查摄像头是否正常工作。")
            break
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imshow('Captured Frame', frame)
            res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)  # 识别的全部结果
            main_emo = res[0]["dominant_emotion"]  # 主表情
            cons = res[0]["face_confidence"]  # 置信度
            cn_emo2 = emo_dict[main_emo]  # 转中文
            if cons < 0.1:
                print("\033[91m置信度过低，再拍一张。\033[0m")
            print(f"此时，你的表情有{cons}的概率是{cn_emo2}。")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return cn_emo2


def VidLocal():
    video_path = input("请输入视频路径(路径不需用引号包裹)：")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频文件打开失败！")
        exit()

    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"分析过程中可能出现的错误: {e}")

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rand_emo = random.choice(emo_list)
    print(f"请做出与{rand_emo}相反的表情。")
    a = input("输入1 打开本地图片检测表情\n"
              "输入2 打开摄像头检测表情\n"
              "输入3 打开本地视频检测表情\n"
              "请选择输入：")
    if a == "1":
        if rand_emo != ImgLocal():
            print(f"恭喜你，你成功了！你做出了与{rand_emo}相反的表情。")
        else:
            print("失败了，你做出了与" + rand_emo + "相同的表情。")
    elif a == "2":
        if rand_emo != VidCap():
            print(f"恭喜你，你成功了！你做出了与{rand_emo}相反的表情。")
        else:
            print("失败了，你做出了与" + rand_emo + "相同的表情。")
    elif a == "3":
        VidLocal()
