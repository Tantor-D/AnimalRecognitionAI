import os

import predictLabel

if __name__ == "__main__":
    pic_path = "D:\Software_data\Pycharm_prj\AnimalRecognitionAI\img"

    img_path_list = os.listdir(pic_path)

    results = []
    rightNum = 0
    falseNum = 0
    for img_path in img_path_list:
        print(os.path.join(pic_path, img_path))
        predictAns = predictLabel.predictLabel(os.path.join(pic_path, img_path), "local")
        results.append(img_path + "  " + str(predictAns + 1))
        if ((predictAns + 1) == int((img_path.split('_'))[0])):
            rightNum = rightNum + 1
        else:
            falseNum = falseNum + 1

    for result in results:
        print(result)
    print(f"right:{rightNum}    false:{falseNum}")