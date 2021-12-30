# -*- coding: utf-8 -*-
"""
@author: Theo-eBrown
"""
import os
import time
from string import ascii_letters,digits
import random
import csv
import cv2
import pyautogui
import keyboard
import numpy as np

#initialize version 0.0.8

# will work and save data to the location of the package / module

package_path = "\\".join(os.path.abspath(__file__).split("\\")[:-1])+"\\"

def __help__():
    """help for scrollthescroll"""
    print("\n",
          "|--------------------------------","\n","| scrollthescroll.Prototype","\n","|","\n",
          "|    Prototype Functions","\n","|","\n",
          "|    Prototype().run(MINLINES=2,SCROLLMULTIPLIER=1.2,save_dir='data',file_path=package_path)","\n","|","\n",
          "|     // run the scrollthescroll program //","\n","|","\n",
          "|     MINLINES=2 - minimum 'read' lines before scroll","\n","|","\n",
          "|     SCROLLMULTIPLIER=1.2 - multiplier for amount scrolled after each line read","\n","|","\n"
          "|     save_dir='data' - location for data to be saved","\n","|","\n",
          "|     file_path=package_path - path to package","\n","|","\n",
          "|    Prototype().run(file_path=package)","\n","|","\n",
          "|     // test the program will function //","\n","|","\n",
          "|     iterations=1","\n","|","\n",
          "|     file_path=package_path - path to package","\n","|","\n",
          "|--------------------------------","\n","|","\n"
          )

def minmax(array):
    """(x-min)/(max-min)"""
    return (array-array.min())/(array.max()-array.min())


def outliers(array,STANDARD_DEVIATIONS=3):
    """outliers found based on standard_deviations"""
    return np.where(np.logical_or(array>array.mean()+STANDARD_DEVIATIONS*array.std(),
                                  array<array.mean()-STANDARD_DEVIATIONS*array.std()))[0]

def format_package(package,STANDARD_DEVIATIONS=3):
    """removes outliers and minmaxes
    , returns with shape [[packetTimes],[[xs],[ys]]]
    takes package of standard shape [[packet_times],[[x,y],[x,y],...]]"""
    packet_times,packets = package
    #creating numpy arrays for quicker calculation times
    packet_times = np.array(packet_times)
    packets = list(map(np.array,zip(*packets)))
    #finding outliers in data, will find outliers based on x values
    outlier_index = outliers(packets[0],
                             STANDARD_DEVIATIONS=STANDARD_DEVIATIONS)
    #removing outlier packets and corrosponding packet_time to maintain integrity
    packet_times = np.delete(packet_times,outlier_index)
    packets = list(map(lambda elem:np.delete(elem,outlier_index),packets))
    #minmaxing the arrays (array-min)/(max-min)
    packet_times = minmax(packet_times)
    packets = list(map(minmax,packets))
    return [packet_times,packets]

def create_data_directory(save_dir="data",file_path=package_path):
    """creates directory for storing past samples"""
    datapath = os.path.join(file_path,save_dir)
    try:
        next(os.walk(datapath))
    except StopIteration:
        os.mkdir(datapath)

def pupil_contours(roi_eye):
    """finds the ideal contours using regression with average colour and uBound"""
    #will user simple linear regression to find the optimum bounds to find the pupil
    average_colour = np.array(list(zip(*roi_eye))[0]).mean()
    upper_bound = 0.8095823108713642*average_colour-31.97684199744961
    threshold = cv2.inRange(roi_eye,0,upper_bound)
    contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        upper_bound = 0.8095823108713642*average_colour-3.154791417758304
        threshold = cv2.inRange(roi_eye,0,upper_bound)
        contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    elif len(contours) > 1:
        upper_bound = 0.8095823108713642*average_colour-57.153562889459664
        threshold = cv2.inRange(roi_eye,0,upper_bound)
        contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            upper_bound = 0.8095823108713642*average_colour-31.97684199744961
            threshold = cv2.inRange(roi_eye,0,upper_bound)
            contours,_ = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return contours

class PupilFunctions:
    """used for finding the X,y coordinates of the pupil"""
    def __init__(self):
        self.frame = None
        self.gray  = None
        self.load_haars()
        self.eyes = np.array([[0,0,0,0],[0,0,0,0]])
        self.face = np.array([0,0,0,0])
        self.pupils = [None,None]

    def load_haars(self,file_path=package_path):
        """load Haars Classifiers for feature detection"""
        self.face_cascade = cv2.CascadeClassifier(file_path+
            'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(file_path+
            'haarcascade_eye.xml')
        if self.face_cascade.empty():
            raise IOError('Unable to load the face cascade classifier xml file')
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
        """finds pupils with a homemade method"""
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
            contours = pupil_contours(roi_eye)
            areas = list(map(cv2.contourArea,contours))
            if len(areas) > 0:
                # selecting the largest contour as the "pupil"
                max_index = areas.index(max(areas))
                x_mean = int(np.around(contours[max_index][:,0,0].mean()))
                y_mean = int(np.around(contours[max_index][:,0,1].mean()))
                self.pupils[indx] = (x_mean+face_x+eye_x,y_mean+face_y+eye_y)
                if display:
                    #drawing a circle on the users pupil
                    cv2.circle(self.frame,
                               (face_x+eye_x+x_mean,face_y+eye_y+int(eye_h*0.25)+y_mean),
                               radius=0, color=(0, 255, 0 ), thickness=3)
        return self.pupils

def load_reading_rates(file_path=package_path):
    """loading reading_rates"""
    with open(file_path+"reading_rates.txt","r") as file:
        reading_rates = np.array(list(map(float,file.read().split("|")[1:])))
    reading_rates = np.sort(reading_rates,axis=0)
    reading_std = reading_rates.std()
    return reading_rates,reading_std

def clean_package(formatted_package):
    """removes head, tail and outliers"""
    packet_times,packets = formatted_package
    x_packets = packets[0]
    try:
        # finding the areas that have a +ve gradient, evident of the tail / head
        index = np.argwhere(x_packets[1::2]-x_packets[::2]>0)
    except ValueError:
        index = np.argwhere(x_packets[1::2]-np.flip(x_packets[-3::-2])>0)
    index  = np.concatenate((index*2,(index)*2+1),axis=1).flatten()
    #^ getting list of all positions of +ve gradient
    index = np.split(index,np.where(np.diff(index)!=1)[0]+1)
    packet_times = np.delete(packet_times,index[-1])#remove the last section of +ve gradient (tail)
    packets = list(map(lambda packet:np.delete(packet,index[-1]),packets))
    try:
        if index[0][0] == 0:
            packet_times = np.delete(packet_times,index[0])
            packets = list(map(lambda packet:np.delete(packet,index[0]),packets))
    except IndexError:
        pass
    return [packet_times,packets]

class PacketModel:
    """model responsible for finding when user has read a line"""
    def __init__(self):
        self.regression_functions = RegressionFunctions()

        self.package = [[],[]]
        self.x_displacement = 0
        self.reading_amount = 0

        self.reading_rates,self.reading_std = load_reading_rates()

        self.word_count = 0

    def build_package(self,packet_time,packet,DX_THRESHOLD=0.3):
        """for deciding when a line had been 'read'"""
        self.package[0].append(packet_time)
        self.package[1].append(packet)
        #packages have a static minimum size, keeps minmax effective
        if len(self.package[0]) > 10:
            #time_change used to ensure the package is a continious stream of reading
            time_change = self.package[0][-1]-self.package[0][-2]
            # checking if the pupil is moving from right to left - changing line
            if self.package[1][-1][0]-self.package[1][-2][0] > 0 or time_change > 2:
                formatted_package = format_package(self.package)
                self.x_displacement += formatted_package[1][0][-1]-formatted_package[1][0][-2]
                #changing must be above some threshold to ensure it is meaningful action
                if self.x_displacement > DX_THRESHOLD:
                    # removing the head and tail of package for accurate regression const
                    cleaned_package = clean_package(formatted_package)
                    # if user is reading package should have negative correlation
                    if self.r_neuron(cleaned_package):
                        # if user has 'read' for enough will scroll a line
                        if self.t_neuron(self.package):
                            # resetting attributes and returning package
                            package = self.package[:]
                            self.package = [[],[]]
                            self.x_displacement = 0
                            return package
                        self.package = [[],[]]
                        self.x_displacement = 0
                        return None
                    self.package = [[],[]]
                    self.x_displacement = 0
                    return None
            elif self.x_displacement > 0:
                formatted_package = format_package(self.package)
                self.x_displacement += formatted_package[1][0][-1]-formatted_package[1][0][-2]
                self.x_displacement = max(self.x_displacement,0)
        return None

    def r_neuron(self,cleaned_package,r_THRESHOLD=-0.7):
        """requires cleanned package, r_THRESHOLD=rejection threshold for regression
        constant"""
        xaxis,packets = cleaned_package
        yaxis = packets[0]
        if len(xaxis)>3:
            _,regression_constant = self.regression_functions.fit(xaxis,yaxis)
            if regression_constant < r_THRESHOLD:
                return True
        return None

    def t_neuron(self,package,STANDARD_DEVIATIONS=3.0):
        """determines whether enough time has elapsed for user to have read line,
        STANDARD_DEVIATIONS: standard deviations for estimate reading time,
        MIN_WORDS : min words for line to be auto scrolled"""
        packet_times,_ = package
        self.reading_amount += (packet_times[-2]-packet_times[0])
        minimum_reading_time = self.word_count/(self.reading_rates[-1]
                                                +STANDARD_DEVIATIONS*self.reading_std)
        if self.reading_amount > minimum_reading_time:
            self.reading_amount -= minimum_reading_time
            return True
        return False

class RegressionFunctions:
    """linear regression functions"""
    def __init__(self):
        self.gradient = None
        self.y_intercept = None
        self.regression_constant = None

    def fit(self,X,y):
        """will apply linear regression and return constants"""
        xY = (X*y).sum()
        Y  = y.sum()
        x  = X.sum()
        xs = (X**2).sum()
        var = len(X)*xs-x**2
        self.gradient = (len(X)*xY-x*Y)/var
        self.y_intercept = (Y*xs-x*xY)/var
        self.regression_constant = (len(X)*xY-x*Y)/np.sqrt(var*(len(X)*(y**2).sum()-Y**2))
        return (self.gradient,self.y_intercept),self.regression_constant

    def Predict(self,X):
        """y = mX+c"""
        return X*self.gradient+self.y_intercept

def save_package(package,save_dir="data",file_path=package_path):
    """"saving a prototype run for analysis, saves the package to .csv file"""
    #will need to assign a random hash to the files so that there are no clashes of data in time
    datapath = os.path.join(file_path,save_dir,"")
    chars = ascii_letters+digits
    file_name = "S{}.csv".format("".join([random.choice(chars) for i in range(15)]))
    datapath = os.path.join(datapath,file_name)
    with open(datapath,"w",encoding="UTF8") as file:
        csv_writer = csv.writer(file)
        header = ["T","Lx","Ly","Rx","Ry"]
        csv_writer.writerow(header)
        for row in list(zip(*package)):
            csv_writer.writerow(row)
# v.0.0.9:

def seperate_words(line):
    """seperate words in a line"""
    index = np.argwhere(line>0)
    index = np.unique(index[:,1])
    difference = index[1:]-index[:-1]
    difference = np.split(index,np.where(difference>6)[0]+1)
    if len(difference) > 0:
        words = [line[:,dif[0]-1:dif[-1]+1] for dif in difference if len(dif) > 0]
    return words

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

    def filter_frame(self):
        """cut and filter frame"""
        #fframe = filtered frame, who cares about naming conventions?
        self.fframe = self.frame[int(self.screen_height*0.1):,
                                 int(self.screen_width*0.05):-int(self.screen_width*0.05)]
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
                                   if len(dif) > 0],dtype=object)
            if whitespace:
                self.lines[0] = np.concatenate((self.fframe[0:index[0]],
                                                self.lines[0])) #include top whitespace
        return self.lines

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
            word_count = map(lambda elem:len(seperate_words(elem)),lines)
            index = map(lambda elem:elem.shape[0],lines)
            return list(zip(word_count,index))
        return None

class Prototype:
    """running the current prototype"""
    def __init__(self):
        self.pupil_functions = PupilFunctions()
        self.left_packet_model = PacketModel()
        self.right_packet_model= PacketModel()
        self.screen_functions = ScreenFunctions()
        self.directory = None

    def run_new(self,MINLINES=2,SCROLLMULTIPLIER=1.2,save_dir="data",file_path=package_path):
        """MIN_LINES:minimum amount of lines read before scrolling,
           SCROLLMULTIPLIER: mult for amount scrolled after each line read,
           save_dir: directory to save sample to,
           file_path: where directory should be found."""
        line_count = 0
        save_data = [[],[],[],[],[]]
        start_time = time.perf_counter()
        analysis_time = time.perf_counter()
        print("preparing webcam, please wait ...")
        cap = cv2.VideoCapture(0)
        print("ready, <press space>")
        while True:
            if keyboard.is_pressed("space"):
                break
        while True:
            if time.perf_counter()-analysis_time > 0.5: # analyse screen every 0.5s
                analysis = self.screen_functions.analyse_lines()
                try:
                    self.left_packet_model.word_count,_ = analysis[line_count]
                    self.right_packet_model.word_count,_ = analysis[line_count]
                    analysis_time = time.perf_counter()
                except TypeError:
                    continue # will loop until the reader is reading something
            _,frame = cap.read()
            left_pupil,right_pupil = self.pupil_functions.find_pupils(frame)
            if left_pupil:
                left_package = self.left_packet_model.build_package(time.perf_counter()-start_time,
                                                                    left_pupil)
                if left_package:
                    line_count += 1 # reader has read a line
                    if line_count >= MINLINES:
                        analysis = self.screen_functions.analyse_lines()
                        _,analysis = list(zip(*analysis[:line_count]))
                        pyautogui.scroll(-int(sum(analysis)*SCROLLMULTIPLIER)) #scrolling
                        line_count = 0
                        self.right_packet_model.package = [[],[]]
                        self.right_packet_model.reading_amount=self.left_packet_model.reading_amount
                        try:
                            analysis = self.screen_functions.analyse_lines()
                            self.left_packet_model.word_count,_ = analysis[line_count]
                            self.right_packet_model.word_count,_ = analysis[line_count]
                        except TypeError:
                            pass
            if right_pupil:
                right_package= self.right_packet_model.build_package(time.perf_counter()-start_time,
                                                                     right_pupil)
                if right_package:
                    line_count += 1
                    if line_count >= MINLINES:
                        analysis = self.screen_functions.analyse_lines()
                        _,analysis = list(zip(*analysis[:line_count]))
                        pyautogui.scroll(-int(sum(analysis)*SCROLLMULTIPLIER))
                        line_count = 0
                        self.left_packet_model.package = [[],[]]
                        self.left_packet_model.reading_amount=self.right_packet_model.reading_amount
                        try:
                            analysis = self.screen_functions.analyse_lines()
                            self.right_packet_model.word_count,_ = analysis[line_count]
                            self.left_packet_model.word_count,_ = analysis[line_count]
                        except TypeError:
                            pass
            if right_pupil and left_pupil:
                save_data[0].append(time.perf_counter()-start_time)
                save_data[1].append(left_pupil[0])
                save_data[2].append(left_pupil[1])
                save_data[3].append(right_pupil[0])
                save_data[4].append(right_pupil[1])
            if keyboard.is_pressed("space"):
                break
            cv2.imshow("frame",frame)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        create_data_directory(save_dir=save_dir,file_path=file_path)
        print("saving data, please don't shut down ...")
        save_package(save_data,save_dir=save_dir,file_path=file_path)
        print("data saved")
