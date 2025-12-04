import cv2
import numpy as np

def find_white_circle_info(image_path, show_result=False):
    """
    从黑白二值图里找出最大的白色圆形，返回 (cx, cy, r)。
    如果没有找到，返回 (None, None, None)。
    """
    # 读图并确保是单通道
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    # 二值化（确保只有 0 和 255）
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 找轮廓
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # 按面积排序，取最大的
    largest = max(contours, key=cv2.contourArea)

    # 最小外接圆
    (cx, cy), r = cv2.minEnclosingCircle(largest)

    # 可选：画出来看看
    if show_result:
        vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
        cv2.imshow("result", vis)
        cv2.waitKey(0)

    return cx, cy, r


if __name__ == "__main__":
    cx, cy, r = find_white_circle_info("012736-20221104@083808-R1.png", show_result=True)
    print(f"圆心: ({cx:.2f}, {cy:.2f})  半径: {r:.2f}")