import matplotlib.pyplot as plt
import gif
from config import FEATURE_POINT_SIZE, POSE_POINT_SIZE, HAND_POINT_SIZE, FACE_POINT_SIZE


def showimg(keypoint_result, suf, png_frame=False):
    @gif.frame
    def plot(i):
        plt.axis("off")
        plt.xlim((0, 1))
        plt.ylim((1, 0))
        for j in range(FEATURE_POINT_SIZE):
            if 0 <= j < POSE_POINT_SIZE:
                plt.scatter(keypoint_result[i][j][0], keypoint_result[i][j][1],
                            color=(0, 0, 0.9), s=3)
            elif POSE_POINT_SIZE <= j < POSE_POINT_SIZE + FACE_POINT_SIZE:
                plt.scatter(keypoint_result[i][j][0], keypoint_result[i][j][1],
                            color=(0.9, 0, 0), s=3)
            else:
                plt.scatter(keypoint_result[i][j][0], keypoint_result[i][j][1],
                            color=(0, 0.9, 0), s=3)

        plt.plot([keypoint_result[i][6][0], keypoint_result[i][8][0]],
                 [keypoint_result[i][6][1], keypoint_result[i][8][1]], color=(1, 0, 0))
        # plt.plot([keypoint_result[i][8][0], keypoint_result[i][10][0]],
        #         [keypoint_result[i][8][1], keypoint_result[i][10][1]], color=(0.8, 0, 0))
        plt.plot([keypoint_result[i][8][0], keypoint_result[i][58][0]],
                 [keypoint_result[i][8][1], keypoint_result[i][58][1]], color=(0.8, 0, 0))
        plt.plot([keypoint_result[i][6][0], keypoint_result[i][5][0]],
                 [keypoint_result[i][6][1], keypoint_result[i][5][1]], color=(0, 0.6, 0))
        plt.plot([keypoint_result[i][5][0], keypoint_result[i][7][0]],
                 [keypoint_result[i][5][1], keypoint_result[i][7][1]], color=(0.8, 0.8, 0))
        # plt.plot([keypoint_result[i][7][0], keypoint_result[i][9][0]],
        #         [keypoint_result[i][7][1], keypoint_result[i][9][1]], color=(0, 0.8, 0))
        plt.plot([keypoint_result[i][7][0], keypoint_result[i][37][0]],
                 [keypoint_result[i][7][1], keypoint_result[i][37][1]], color=(0, 0.8, 0))

        plt.plot([keypoint_result[i][37][0], keypoint_result[i][38][0]],
                 [keypoint_result[i][37][1], keypoint_result[i][38][1]], color=(0, 1, 1))
        plt.plot([keypoint_result[i][38][0], keypoint_result[i][39][0]],
                 [keypoint_result[i][38][1], keypoint_result[i][39][1]], color=(0, 1, 1))
        plt.plot([keypoint_result[i][39][0], keypoint_result[i][40][0]],
                 [keypoint_result[i][39][1], keypoint_result[i][40][1]], color=(0, 1, 1))
        plt.plot([keypoint_result[i][40][0], keypoint_result[i][41][0]],
                 [keypoint_result[i][40][1], keypoint_result[i][41][1]], color=(0, 1, 1))

        plt.plot([keypoint_result[i][37][0], keypoint_result[i][42][0]],
                 [keypoint_result[i][37][1], keypoint_result[i][42][1]], color=(0, 0.8, 0.8))
        plt.plot([keypoint_result[i][42][0], keypoint_result[i][43][0]],
                 [keypoint_result[i][42][1], keypoint_result[i][43][1]], color=(0, 0.8, 0.8))
        plt.plot([keypoint_result[i][43][0], keypoint_result[i][44][0]],
                 [keypoint_result[i][43][1], keypoint_result[i][44][1]], color=(0, 0.8, 0.8))
        plt.plot([keypoint_result[i][44][0], keypoint_result[i][45][0]],
                 [keypoint_result[i][44][1], keypoint_result[i][45][1]], color=(0, 0.8, 0.8))

        plt.plot([keypoint_result[i][37][0], keypoint_result[i][46][0]],
                 [keypoint_result[i][37][1], keypoint_result[i][46][1]], color=(0, 0.6, 0.6))
        plt.plot([keypoint_result[i][46][0], keypoint_result[i][47][0]],
                 [keypoint_result[i][46][1], keypoint_result[i][47][1]], color=(0, 0.6, 0.6))
        plt.plot([keypoint_result[i][47][0], keypoint_result[i][48][0]],
                 [keypoint_result[i][47][1], keypoint_result[i][48][1]], color=(0, 0.6, 0.6))
        plt.plot([keypoint_result[i][48][0], keypoint_result[i][49][0]],
                 [keypoint_result[i][48][1], keypoint_result[i][49][1]], color=(0, 0.6, 0.6))

        plt.plot([keypoint_result[i][37][0], keypoint_result[i][50][0]],
                 [keypoint_result[i][37][1], keypoint_result[i][50][1]], color=(0, 0.4, 0.4))
        plt.plot([keypoint_result[i][50][0], keypoint_result[i][51][0]],
                 [keypoint_result[i][50][1], keypoint_result[i][51][1]], color=(0, 0.4, 0.4))
        plt.plot([keypoint_result[i][51][0], keypoint_result[i][52][0]],
                 [keypoint_result[i][51][1], keypoint_result[i][52][1]], color=(0, 0.4, 0.4))
        plt.plot([keypoint_result[i][52][0], keypoint_result[i][53][0]],
                 [keypoint_result[i][52][1], keypoint_result[i][53][1]], color=(0, 0.4, 0.4))

        plt.plot([keypoint_result[i][37][0], keypoint_result[i][54][0]],
                 [keypoint_result[i][37][1], keypoint_result[i][54][1]], color=(0, 0.2, 0.2))
        plt.plot([keypoint_result[i][54][0], keypoint_result[i][55][0]],
                 [keypoint_result[i][54][1], keypoint_result[i][55][1]], color=(0, 0.2, 0.2))
        plt.plot([keypoint_result[i][55][0], keypoint_result[i][56][0]],
                 [keypoint_result[i][55][1], keypoint_result[i][56][1]], color=(0, 0.2, 0.2))
        plt.plot([keypoint_result[i][56][0], keypoint_result[i][57][0]],
                 [keypoint_result[i][56][1], keypoint_result[i][57][1]], color=(0, 0.2, 0.2))

        plt.plot([keypoint_result[i][58][0], keypoint_result[i][59][0]],
                 [keypoint_result[i][58][1], keypoint_result[i][59][1]], color=(1, 0, 1))
        plt.plot([keypoint_result[i][59][0], keypoint_result[i][60][0]],
                 [keypoint_result[i][59][1], keypoint_result[i][60][1]], color=(1, 0, 1))
        plt.plot([keypoint_result[i][60][0], keypoint_result[i][61][0]],
                 [keypoint_result[i][60][1], keypoint_result[i][61][1]], color=(1, 0, 1))
        plt.plot([keypoint_result[i][61][0], keypoint_result[i][62][0]],
                 [keypoint_result[i][61][1], keypoint_result[i][62][1]], color=(1, 0, 1))

        plt.plot([keypoint_result[i][58][0], keypoint_result[i][63][0]],
                 [keypoint_result[i][58][1], keypoint_result[i][63][1]], color=(0.8, 0, 0.8))
        plt.plot([keypoint_result[i][63][0], keypoint_result[i][64][0]],
                 [keypoint_result[i][63][1], keypoint_result[i][64][1]], color=(0.8, 0, 0.8))
        plt.plot([keypoint_result[i][64][0], keypoint_result[i][65][0]],
                 [keypoint_result[i][64][1], keypoint_result[i][65][1]], color=(0.8, 0, 0.8))
        plt.plot([keypoint_result[i][65][0], keypoint_result[i][66][0]],
                 [keypoint_result[i][65][1], keypoint_result[i][66][1]], color=(0.8, 0, 0.8))

        plt.plot([keypoint_result[i][58][0], keypoint_result[i][67][0]],
                 [keypoint_result[i][58][1], keypoint_result[i][67][1]], color=(0.6, 0, 0.6))
        plt.plot([keypoint_result[i][67][0], keypoint_result[i][68][0]],
                 [keypoint_result[i][67][1], keypoint_result[i][68][1]], color=(0.6, 0, 0.6))
        plt.plot([keypoint_result[i][68][0], keypoint_result[i][69][0]],
                 [keypoint_result[i][68][1], keypoint_result[i][69][1]], color=(0.6, 0, 0.6))
        plt.plot([keypoint_result[i][69][0], keypoint_result[i][70][0]],
                 [keypoint_result[i][69][1], keypoint_result[i][70][1]], color=(0.6, 0, 0.6))

        plt.plot([keypoint_result[i][58][0], keypoint_result[i][71][0]],
                 [keypoint_result[i][58][1], keypoint_result[i][71][1]], color=(0.4, 0, 0.4))
        plt.plot([keypoint_result[i][71][0], keypoint_result[i][72][0]],
                 [keypoint_result[i][71][1], keypoint_result[i][72][1]], color=(0.4, 0, 0.4))
        plt.plot([keypoint_result[i][72][0], keypoint_result[i][73][0]],
                 [keypoint_result[i][72][1], keypoint_result[i][73][1]], color=(0.4, 0, 0.4))
        plt.plot([keypoint_result[i][73][0], keypoint_result[i][74][0]],
                 [keypoint_result[i][73][1], keypoint_result[i][74][1]], color=(0.4, 0, 0.4))

        plt.plot([keypoint_result[i][58][0], keypoint_result[i][75][0]],
                 [keypoint_result[i][58][1], keypoint_result[i][75][1]], color=(0.2, 0, 0.2))
        plt.plot([keypoint_result[i][75][0], keypoint_result[i][76][0]],
                 [keypoint_result[i][75][1], keypoint_result[i][76][1]], color=(0.2, 0, 0.2))
        plt.plot([keypoint_result[i][76][0], keypoint_result[i][77][0]],
                 [keypoint_result[i][76][1], keypoint_result[i][77][1]], color=(0.2, 0, 0.2))
        plt.plot([keypoint_result[i][77][0], keypoint_result[i][78][0]],
                 [keypoint_result[i][77][1], keypoint_result[i][78][1]], color=(0.2, 0, 0.2))

        if png_frame:
            plt.savefig(suf + "frame" + str(i) + ".png")

    frames = []
    for i in range(len(keypoint_result)):
        frame = plot(i)
        frames.append(frame)
    gif.save(frames, suf + ".gif")
    print("SAVED:" + suf + ".gif")
