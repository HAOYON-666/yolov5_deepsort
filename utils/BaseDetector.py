from tracker import update_tracker
import cv2


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self,im,filename):

        retDict = {
            'frame': None,
            'news': None,
            'list_of_ids': None,
            'new_bboxes': []
        }
        self.frameCounter += 1


        im, news, new_bboxes = update_tracker(self, im, filename)

        retDict['frame'] = im
        retDict['news'] = news
        retDict['new_bboxes'] = new_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
