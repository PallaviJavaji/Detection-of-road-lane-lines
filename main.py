import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(255, 0, 0), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lane_lines(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 50, 150)

    height, width = edges.shape
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    masked_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )

    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    result_img = cv2.addWeighted(img, 0.8, line_img, 1, 0.0)
    return result_img

# Example usage
input_img = cv2.imread('road_image.jpg')
result = detect_lane_lines(input_img)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
