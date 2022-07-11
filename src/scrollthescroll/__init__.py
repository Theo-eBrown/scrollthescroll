# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:16:36 2022

@author: PeterB
"""

import os
import time
# from string import ascii_letters,digits
# import random
# import csv
import cv2
import pyautogui
import keyboard
import numpy as np

package_path = "\\".join(os.path.abspath(__file__).split("\\")[:-1])+"\\" #only works on windows

def get_pupil_position(eye):
    """returns x position of pupil relative to the left of the eye"""
    pupil_y_pos = eye.shape[0]//2
    middle = eye[pupil_y_pos,:] #get middle row of eye
    pupil_x_pos = int(np.argwhere(middle==np.min(middle))[0]) #return the position of darkest point
    return (pupil_x_pos,eye.shape[0]//2)

class PupilFunctions:
    """used for finding the X,y coordinates of the pupil"""
    def __init__(self):
        self.frame = None
        self.gray  = None
        self.eyes = np.array([[0,0,0,0],[0,0,0,0]])
        self.face = np.array([0,0,0,0])
        self.pupils = [None,None]

        self.face_cascade = self.eye_cascade = None

    def load_haars(self,file_path=package_path):
        """load Haars Classifiers for feature detection"""
        self.face_cascade = cv2.CascadeClassifier(file_path+
            'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(file_path+
            'haarcascade_eye.xml')
        if self.face_cascade.empty():
            raise IOError(f'Unable to load the face cascade classifier xml file{file_path+"haarcascade_eye.xml"}')
        if self.eye_cascade.empty():
            raise IOError('Unable to load the eye cascade classifier xml file')

    def set_frame(self,frame):
        """get frame from webcam"""
        self.frame = frame
        self.gray  = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

    def find_face(self,display=True,scalefactor=1.1):
        """finds face with haars classifier"""
        #the smaller the scalefactor the more reliable the algorithm is
        # smaller scalefactors increase the amount of time searching
        faces = self.face_cascade.detectMultiScale(self.gray,scalefactor,5)
        if len(faces) > 0:
            self.face = sorted(faces,reverse=True,key=lambda elem:elem[2]*elem[3])[0]
            if display:
                (face_x,face_y,face_w,face_h) = self.face
                cv2.rectangle(self.frame, (face_x,face_y),
                              (face_x+face_w,face_y+(face_h//2)), (0,255,0), 3)
            return self.face
        self.face = np.array([0,0,0,0]) # prevent an error from being thrown?
        return self.face

    def find_eyes(self,display=True,scalefactor=1.1):
        """finds eyes with haars classifier"""
        (face_x,face_y,face_w,face_h) = self.face
        face_area = self.gray[face_y:face_y+face_h//2,face_x:face_x+face_w]
        eyes = self.eye_cascade.detectMultiScale(face_area,scalefactor)
        self.eyes = np.array([[0,0,0,0],[0,0,0,0]])
        for (eye_x,eye_y,eye_w,eye_h) in eyes:
            if eye_x > face_w/2:
                self.eyes[1] = (eye_x,eye_y,eye_w,eye_h)
            else:
                self.eyes[0] = (eye_x,eye_y,eye_w,eye_h)
            if display:
                cv2.rectangle(self.frame,(face_x+eye_x,face_y+eye_y),
                              (face_x+eye_x+eye_w,face_y+eye_y+eye_h), (0,255,0), 3)
        return self.eyes

    def find_pupils(self,frame,display=True,scalefactor=1.1):
        """finds pupils x and y position"""
        # edit the algorithm
        self.set_frame(frame)
        self.find_face(display=display,scalefactor=scalefactor)
        self.find_eyes(display=display,scalefactor=scalefactor)
        self.pupils = [None,None]
        (face_x,face_y,face_w,face_h) = self.face
        face_area = self.gray[face_y:face_y+face_h//2,face_x:face_x+face_w]
        for indx,(eye_x,eye_y,eye_w,eye_h) in enumerate(self.eyes):
            if not any((eye_x,eye_y,eye_w,eye_h)):
                continue
            roi_eye = face_area[(eye_y+int(eye_h*0.25)):(eye_y+eye_h-int(eye_h*0.2)),
                                eye_x:eye_x+eye_w]
            #removing any holes in the eye contour and rounding the shape off for better estimation
            roi_eye = cv2.morphologyEx(roi_eye, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            roi_eye = cv2.morphologyEx(roi_eye, cv2.MORPH_ERODE, np.ones((2,2),np.uint8))
            roi_eye = cv2.morphologyEx(roi_eye, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
            pupil_x,pupil_y = get_pupil_position(roi_eye)
            self.pupils[indx] =(pupil_x+face_x+eye_x, pupil_y+face_y+eye_y)
            if display:
                #drawing a circle on the users pupil
                cv2.circle(self.frame,
                           (face_x+eye_x+pupil_x,face_y+eye_y+int(eye_h*0.25)+pupil_y),
                           radius=0, color=(0,255,0), thickness=3)
        return self.pupils

# relies on the words having a certain seperation of pixels
def seperate_words(line):
    """seperate words in a line"""
    height,_ = line.shape
    line = line[int(height*0.2):-int(height*0.2)] # cut top and bottom of other lines
    index = np.argwhere(line>0)
    index = np.unique(index[:,1])
    difference = index[1:]-index[:-1]
    difference = np.split(index,np.where(difference>6)[0]+1) # if 'gap' between black pixels is enough will seperate a word
    if len(difference) > 0:
        words = [line[:,dif[0]-1:dif[-1]+1] for dif in difference if len(dif) > 0]
    line_size = index.max() - index.min() if len(index) else 0
    return words, line_size

# screenFunctions: replacing having to manually load a pdf.
# will take a screen shot of the screen and will find lines on the page (requires the user to have ebook loaded)
# need to redo this code such that all screenshots are resized to the same size so that the word seperation technique is more robust
# take screenshot
# take desired area of screenshot
# resize and then filter screenshot
# analyse the screenshot
class ScreenFunctions:
    """replaces the use of libraries for reading the page (replacing extract functions),
    will have a better understanding / control of the scroll"""
    def __init__(self):
        self.frame = None
        self.fframe = None
        self.lines = np.array([])
        self.screen_height,self.screen_width = np.array(pyautogui.screenshot()).shape[:2]

    def set_frame(self):
        """take screenshot, filter to appropriate area and convert to cv2 format"""
        self.frame = np.array(pyautogui.screenshot())
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        return self.frame

    # filter_frame: filter frame and binarise
    # filter will work to seperate words and lines better, such that individual lines and words can be detected
    # binary filter also used to ease the acquisition of lines (1 = black, 0 = white)
    def filter_frame(self):
        """cut and filter frame"""
        #fframe = filtered frame, who cares about naming conventions?
        self.fframe = self.frame[int(self.screen_height*0.1):,
                                 int(self.screen_width*0.05):-int(self.screen_width*0.05)] # get the appropriate area of the screen
        #otsu threshold:
        _,self.fframe = cv2.threshold(self.fframe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.fframe = cv2.inRange(self.fframe,0,254) # binarise frame
        return self.fframe

    def seperate_lines(self, whitespace=False):
        """seperate frame into lines of writing"""
        index = np.argwhere(self.fframe>0)
        index = np.unique(index[:,0])
        difference = index[1:]-index[:-1]
        if not whitespace:
            difference = np.split(index,np.where(difference>1)[0]+1)
        else:
            difference = np.split(index,np.where(difference>1)[0]) # include whitespace above line
        if len(difference) > 1: #avoid throwing an IndexError
            self.lines = np.array([self.fframe[dif[0]-1:dif[-1]+1] for dif in difference
                                   if len(dif) > 0],dtype=object) #if seperation of > 1 pixel between rows will seperate
            if whitespace:
                self.lines[0] = np.concatenate((self.fframe[0:index[0]],
                                                self.lines[0])) #include top whitespace
        return self.lines

    # analyse_lines for looking at the word count on a line, line will be found using the seperate_lines() function
    # analyse_lines doesn't apply any more filters to the image, will simply look for gaps in black and decide to seperate lines
    # will also return the height of the line, so that the exact height of a line can be scrolled once it has been read
    def analyse_lines(self):
        """return height and word count of each line"""
        self.set_frame()
        self.filter_frame()
        self.seperate_lines(whitespace=True)
        if len(self.lines) > 0:
            index = np.array([line.shape[0] for line in self.lines][:-1])
            # choosing the line the user will most likely be reading:
            lines = self.lines[np.argwhere(index>np.floor(index.mean()-index.std()*1.5))[:,0]]
            # ^ removing half covered lines / not full lines
            word_count = list(map(seperate_words,lines))
            index = list(map(lambda elem:elem.shape[0],lines)) #gives the lines height
            return [(len(word_count),(width,height)) for (count,width),height in zip(word_count,index)]
        # return word_count,(shape of line)
        return None

def clean_package(package):
    """will remove the head and tail of a package; +ve gradient at start and end of the package"""
    packet_times,packets = package
    packet_times = np.array(packet_times)
    packets = np.array(packets)
    x_packets = packets[:,0]
    grad_index = np.argwhere(x_packets[1:] - x_packets[:-1] > 0) #positions of +ve gradient
    if len(grad_index) <= 1: #bug fix will not work if there is only one point of +ve gradient
        return [packet_times,packets]
    ind_index = np.where(grad_index[1:] - grad_index[:-1] > 2)[0] #'groups' of +ve gradient
    ind_index = [0] if not len(ind_index) else ind_index #if there is only one 'group' of +ve gradient at end
    packet_times = packet_times[:grad_index[ind_index[-1]+1:][0][0]+1]#removing the tail of package
    packets = packets[:grad_index[ind_index[-1]+1:][0][0]+1]
    if 0 in grad_index[:ind_index[0]]:
        packet_times = packet_times[grad_index[:ind_index[0]][-1][0]+1:]
        packets = packets[grad_index[:ind_index[0]][-1][0]+1:]
    return [packet_times,packets]

def reg_coef(x,y):
    """pearson regression coef, x and y one dimensional and numpy arrays"""
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x-x_mean)*(y-y_mean))
    denom = np.sqrt(np.sum((x-x_mean)**2)*np.sum((y-y_mean)**2))
    return num/denom

class NewPackageModel:
    def __init__(self):
        """new package algorithm"""
        self.package = [[],[]]
        self.time_read = 0
        
        self.reading_rate = 3.00 # average reading rate of 180 - 320 wpm (for fiction), using lower # ref Google
        self.word_count = None
    
    def build_package(self,packet_time,packet):
        """deciding when a line is 'read'"""
        self.package[0].append(packet_time)
        self.package[1].append(packet)
        if len(self.package[0]) >= 10:
            if self.difference_check():
                if self.regression_check():
                    if self.time_check():
                        return self.package
        return None

    def difference_check(self):
        """checking whether the eye has moved to the left sufficiently, first check"""
        if self.package[1][-1][0] - self.package[1][-2][0] > 0: #positive gradient - pupil moved to the left
            packet = np.array(self.package[1])[:,0] # all eye x positions
            packet_x_max = packet.max()
            packet_range = packet_x_max - packet.min()
            if packet_x_max - packet_range*0.33 < self.package[1][-1][0]: #if eye returns to ~ starting pos
                return True
            try:
                if not self.regression_check():
                    self.package = [[],[]]
            except IndexError: # error thrown that cannot quite pin down
                pass
        return None

    def regression_check(self, r_threshold = -0.7):
        """checking for a strong negative correlation, uses Pearson's Regression Coefficient"""
        packet_times,packets = clean_package(self.package)
        r_coef = reg_coef(packet_times[:],packets[:,0][:])
        if r_coef < r_threshold and len(packet_times > 4):
            return True
        return False

    def time_check(self):
        """check whether the user has read enough for a line to be read"""
        packet_times,_ = self.package
        self.time_read += max(packet_times) - min(packet_times) # add the time range of the packet
        minimum_reading_time = self.word_count / self.reading_rate
        if self.time_read > minimum_reading_time:
            self.time_read = 0
            return True
        return None

class Prototype:
    def __init__(self):
        """developing a new algorithm"""
        self.right_packet_model = NewPackageModel()
        self.left_packet_model = NewPackageModel()

        self.pupil_functions = PupilFunctions()
        self.screen_functions = ScreenFunctions()

        self.screen_analysis = None
        self.scroll_lines = 0
    
    def load_line_analysis(self):
        """loading line analysis"""
        screen_width = np.array(pyautogui.screenshot()).shape[1]
        screen_analysis = self.screen_functions.analyse_lines()
        try:
            while True:
                if screen_analysis[self.scroll_lines][1][0] / screen_width > 0.4:
                    self.left_packet_model.word_count,_ = screen_analysis[self.scroll_lines]
                    self.right_packet_model.word_count,_ = screen_analysis[self.scroll_lines]
                    return screen_analysis
                self.scroll_lines += 1
        except:
            return False

    def scroll(self, scroll_multiplier=1.2):
        """will scroll self.scroll_lines lines"""
        analysis = self.screen_functions.analyse_lines()
        analysis = [i for _,(_,i) in analysis[:self.scroll_lines]]
        pyautogui.scroll(-int(sum(analysis)*scroll_multiplier))
        self.scroll_lines = 0

    def run(self, minimum_lines=2, scroll_multiplier = 1.2, file_path=package_path):
        """prototype algorithm"""
        self.pupil_functions.load_haars(file_path=file_path)
        read_lines = 0
        print("preparing webcam, please wait ...")
        cap = cv2.VideoCapture(0)
        print("ready, <press space>")
        while True:
            if keyboard.is_pressed("space"):
                break
        print("<press space to finish>")
        analysis_time = start_time = time.perf_counter()
        while True:
            if time.perf_counter() - analysis_time > 0.5:
                analysis = self.screen_functions.analyse_lines()
                if analysis != self.screen_analysis:
                    if self.load_line_analysis():
                        analysis_time = time.perf_counter()
                    else:
                        continue
                else:
                    analysis_time = time.perf_counter()
            _,frame = cap.read()
            left_pupil,right_pupil = self.pupil_functions.find_pupils(frame)
            if left_pupil:
                left_package = self.left_packet_model.build_package(time.perf_counter()-start_time,
                                                                    left_pupil)
                if left_package:
                    read_lines += 1
                    self.scroll_lines += 1
                    if read_lines >= minimum_lines:
                        print(self.scroll_lines)
                        self.scroll(scroll_multiplier=scroll_multiplier)
                        self.screen_analysis = self.load_line_analysis()
                        read_lines = 0
                        #resetting package variables:
                    self.left_packet_model.package = [[],[]]
                    self.right_packet_model.package = [[],[]]
                    self.right_packet_model.time_read = self.left_packet_model.time_read
                        
            if right_pupil:
                right_package = self.right_packet_model.build_package(time.perf_counter()-start_time,
                                                                      right_pupil)
                if right_package:
                    read_lines += 1
                    self.scroll_lines += 1
                    if read_lines >= minimum_lines:
                        print(self.scroll_lines)
                        self.scroll(scroll_multiplier=scroll_multiplier)
                        self.screen_analysis = self.load_line_analysis()
                        read_lines = 0
                    self.right_packet_model.package = [[],[]]
                    self.left_packet_model.package = [[],[]]
                    self.left_packet_model.time_read = self.right_packet_model.time_read
            cv2.imshow("frame",frame)
            cv2.waitKey(1)
            if keyboard.is_pressed("space"):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = Prototype()
    p.run()
