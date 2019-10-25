
import numpy as np
import matplotlib.pyplot as plt


def get_joint(x, idx):
    px = x[2*idx]
    py = x[2*idx+1]
    return px, py


def set_joint(x, idx, px, py):
    x[2*idx] = px
    x[2*idx+1] = py
    return


def get_an_example_of_standing_skeleton():
    data = [7, 67, 7041, "stand", "stand_03-08-20-24-55-587/00055.jpg", 0.5670731707317073, 0.11005434782608697, 0.5670731707317073, 0.18342391304347827, 0.5182926829268293, 0.1875, 0.5030487804878049, 0.27309782608695654, 0.5030487804878049, 0.34239130434782605, 0.6189024390243902, 0.18342391304347827, 0.6310975609756098, 0.2649456521739131, 0.6310975609756098, 0.3342391304347826, 0.5365853658536586,
            0.34646739130434784, 0.5335365853658537, 0.46467391304347827, 0.5335365853658537, 0.5747282608695652, 0.600609756097561, 0.34646739130434784, 0.600609756097561, 0.4565217391304348, 0.5945121951219512, 0.5665760869565217, 0.5579268292682927, 0.10190217391304347, 0.5762195121951219, 0.09782608695652173, 0.5426829268292683, 0.11005434782608697, 0.5884146341463414, 0.11005434782608697]
    skeleton = np.array(data[5:])
    return skeleton


def get_a_normalized_standing_skeleton():
    x = get_an_example_of_standing_skeleton()

    NECK = 1
    L_THIGH = 8
    R_THIGH = 11

    # Remove offset by setting neck as origin
    x0, y0 = get_joint(x, NECK)
    x[::2] -= x0
    x[1::2] -= y0

    # Scale the skeleton by taking neck-thigh distance as height
    x0, y0 = get_joint(x, NECK)
    _, y11 = get_joint(x, L_THIGH)
    _, y12 = get_joint(x, R_THIGH)
    y1 = (y11 + y12) / 2
    height = abs(y0 - y1)
    x /= height
    return x


def draw_skeleton_joints(skeleton):
    x = skeleton[::2]
    y = skeleton[1::2]
    plt.plot(-x, -y, "r*")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    skeleton = get_a_normalized_standing_skeleton()
    draw_skeleton_joints(skeleton)
