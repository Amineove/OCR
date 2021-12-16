import cv2
import numpy as np
class Trapezoid:

    def __init__(self, img_size, r=5, color=(143, 149, 47),
                bottom_color=(196, 196, 196), border_color=(255, 0, 135)):
        # Initialize the contours
        self.contours = np.array([[[0 + r, 0 + r]],
                               [[0 + r, img_size[1] - r]],
                               [[img_size[0] - r, img_size[1] - r]],
                               [[img_size[0] - r, 0 + r]]])

        # Initialize the radius of the borders
        self.r = r
        # Initialize the colors of the trapezoid
        self.color = color
        self.bottom_color = bottom_color
        self.border_color = border_color

    def get_border_index(self, coord):
        # A border is return if the coordinates are in its radius
        for i, b in enumerate(self.contours[:, 0, :]):
            dist = sum([(b[i] - x) ** 2 for i, x in enumerate(coord)]) ** 0.5
            if  dist < self.r:
                return i
        # If no border, return None
        return None

    def set_border(self, border_index, coord):
        self.contours[border_index, 0, :] = coord

class Scanner():

    def __init__(self, input_path, output_path):
        self.input = cv2.imread(input_path)
        self.output_path = output_path
        # get the shape and size of the input
        self.shape = self.input.shape[:-1]
        self.size = tuple(list(self.shape)[::-1])

        # create a trapezoid to drag and drop and its perspective matrix
        self.M = None
        self.trapezoid = Trapezoid(self.size,
                                r=min(self.shape) // 100 + 2,
                                color=(153, 153, 153),
                                border_color=(255, 0, 136),
                                bottom_color=(143, 149, 47))

        # Initialize the opencv window
        cv2.namedWindow('Rendering', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Rendering', self.drag_and_drop_border)

        # to remember wich border is dragged if exists
        self.border_dragged = None

    def draw_trapezoid(self, img):
        # draw the contours of the trapezoid
        cv2.drawContours(img, [self.trapezoid.contours], -1,
                       self.trapezoid.color, self.trapezoid.r // 3)
        # draw its bottom
        cv2.drawContours(img, [self.trapezoid.contours[1:3]], -1,
                       self.trapezoid.bottom_color, self.trapezoid.r // 3)
        # Draw the border of the trapezoid as circles
        for x, y in self.trapezoid.contours[:, 0, :]:
            cv2.circle(img, (x, y), self.trapezoid.r,
                      self.trapezoid.border_color, cv2.FILLED)
        return img

    def drag_and_drop_border(self, event, x, y, flags, param):
        # If the left click is pressed, get the border to drag
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the selected border if exists
            self.border_dragged = self.trapezoid.get_border_index((x, y))

        # If the mouse is moving while dragging a border, set its new positionAxel THEVENOT
        elif event == cv2.EVENT_MOUSEMOVE and self.border_dragged is not None:
                self.trapezoid.set_border(self.border_dragged, (x, y))

        # If the left click is released
        elif event == cv2.EVENT_LBUTTONUP:
            # Remove from memory the selected border
            self.border_dragged = None

    def actualize_perspective_matrices(self):
        # get the source points (trapezoid)
        src_pts = self.trapezoid.contours[:, 0].astype(np.float32)

        # set the destination points to have the perspective output image
        h, w = self.shape
        dst_pts = np.array([[0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]], dtype="float32")

        # compute the perspective transform matrices
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def run(self):
        while True:
            self.actualize_perspective_matrices()

            # get the output image according to the perspective transformation
            img_output = cv2.warpPerspective(self.input, self.M, self.size)

            # draw current state of the trapezoid
            img_input = self.draw_trapezoid(self.input.copy())
            # Display until the 'Enter' key is pressed
            cv2.imshow('Rendering', np.hstack((img_input, img_output)))
            if cv2.waitKey(1) & 0xFF == 13:
                break

        # Save the image and exit the process
        cv2.imwrite(self.output_path, img_output)
        cv2.destroyAllWindows()
    def bonus(self, img):
        """Make the output better"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = hsv[..., 2]
        value = (value - np.min(value)) / (np.max(value) - np.min(value))
        hsv[..., 2] = (value * 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
sc=Scanner("images/note.jpg","images/scanned-note.jpg")
sc.run()