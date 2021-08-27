# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:45:59 2021

@author: PeterB
"""
import os
import cv2
import time
import pyautogui
import keyboard
import pdfplumber
import subprocess
import numpy as np

"""will work and save data to the location of the package / module"""
package_path = "\\".join(os.path.abspath(__file__).split("\\")[:-1])+"\\"

def scroll_line(nlines=1):
    """will use pyautogui to scroll lines - for 150% PDF"""
    for i in range(nlines):
        for i in range(3):
            pyautogui.scroll(-14)

def scroll_page(npages=1):
    """will us pyautogui to scroll the bit at the end of a page"""
    for i in range(npages):
        for i in range(19):
            pyautogui.scroll(-25)

"""
statistical functions, 
utilizing numpy to transform arrays
"""

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

"""
independent packet functions,
functions for saving and loading any data
functions are not used within the 
"""
def create_data_directory(save_dir="packageSamples",file_path=package_path):
    """creates directory for storing past samples"""
    datapath = os.path.join("data","")
    try:
        next(os.walk(file_path+datapath))
    except StopIteration:
        os.mkdir(file_path+datapath)
    datapath = os.path.join("data",save_dir,"")
    try:
        next(os.walk(file_path+datapath))
    except StopIteration:
        os.mkdir(file_path+datapath)

def load_extract_index(save_dir="packageSamples",file_path=package_path):
    """index for saved samples"""
    extract_index = {}
    def assign_key(elem):
        # for assinging keys for a dictionary
        key,content = elem.split(":")
        extract,page_number = content.split(",")
        extract_index[key] = (extract,int(page_number))
    datapath = os.path.join("data",save_dir,"")
    with open(file_path+datapath+"extract_index.txt","r") as file:
        list(map(assign_key,file.read().split("|")[1:]))
    return extract_index

def load_data(directory_index,package_index,save_dir="packageSamples",file_path=package_path):
    """loading a saved sample in form [[times],[packets]]"""
    #getting the location of the desired data
    datapath = os.path.join("data",save_dir,"")
    _,directories,_ = next(os.walk(file_path+datapath))
    directory = sorted(directories,key=lambda elem:int(elem[7:]))[directory_index]
    datapath = os.path.join("data",save_dir,directory,"")
    _,_,packages = next(os.walk(file_path+datapath))
    package   = sorted(packages,key= lambda elem:int(elem[8:-4]))[package_index]
    #loading from .txt file
    with open(file_path+datapath+package,"r") as file:
        times,packets = file.read().split("|")
    #transferring from string to array
    #packets saved in form [x,y],[x,y],...
    packets = list(map(lambda elem: [float(i) for i in elem.split(",")],packets.split("-")))
    times   = list(map(float,times.split("-")))
    return [times,packets]


"""
pupil functions,
pupil functions work with good level of success however the population that they have been 
tested on is very small and is the most likely part of the prototype to fail - will require 
larger test population or data-set to perfect. (finding eyes and finding face is reliable for it uses
the trusted haars method from cv2 - utilizes larger data population - finding pupils is weak part).
Integral part of run
"""

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
        # however smaller scalefactors increase the amount of time searching
        faces = self.face_cascade.detectMultiScale(self.gray,scalefactor,5)
        if len(faces) > 0:
            self.face = sorted(faces,reverse=True,key=lambda elem:elem[2]*elem[3])[0]
            if display:
                (face_x,face_y,face_w,face_h) = self.face
                cv2.rectangle(self.frame, (face_x,face_y),
                              (face_x+face_w,face_y+(face_h//2)), (0,255,0), 3)
            return self.face
        self.face = np.array([0,0,0,0])
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
                max_index = areas.index(max(areas))
                x_mean = int(np.around(contours[max_index][:,0,0].mean()))
                y_mean = int(np.around(contours[max_index][:,0,1].mean()))
                self.pupils[indx] = (x_mean+face_x+eye_x,y_mean+face_y+eye_y)
                if display:
                    cv2.circle(self.frame,
                               (face_x+eye_x+x_mean,face_y+eye_y+int(eye_h*0.25)+y_mean),
                               radius=0, color=(0, 255, 0 ), thickness=3)
        return self.pupils
"""
packet model is in infancy stages,
works well however is perhaps overactive when the user is not reading.
When reading however model works well

once again suffers from low population size and reading rates are generated from small population
"""
class packet_model:
    """model responsible for finding when user has read a line"""
    def __init__(self):
        self.extract_functions = ExtractFunctions()
        self.regression_functions = RegressionFunctions()

        self.package = [[],[]]
        self.x_displacement = 0
        self.line_index = 0
        self.reading_amount = 0

    def load_extract(self,datapath,page_number,call=True):
        """ease of use, will also reset all attributes"""
        self.extract_functions.load_extract(datapath,call=call)
        self.extract_functions.load_page(page_number-1)

        self.package = [[],[]]
        self.x_displacement = 0
        self.line_index = 0
        self.reading_amount = 0

    def build_package(self,packet_time,packet,DX_THRESHOLD=0.3):
        """for deciding when a line had been 'read'"""
        self.package[0].append(packet_time)
        self.package[1].append(packet)
        #packages have a static minimum size, keeps minmax effective
        if len(self.package[0]) > 10:
            # checking if the pupil is moving from right to left - changing line
            if self.package[1][-1][0]-self.package[1][-2][0] > 0:
                formatted_package = format_package(self.package)
                self.x_displacement += formatted_package[1][0][-1]-formatted_package[1][0][-2]
                #changing must be above some threshold to ensure it is meaningful action
                if self.x_displacement > DX_THRESHOLD:
                    # removing the head and tail of package for accurate regression const
                    cleaned_package = self.clean_package(formatted_package)
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

    def clean_package(self,formatted_package,STANDARD_DEVIATIONS=2):
        """removes head, tail and outliers"""
        packet_times,packets = formatted_package
        x_packets = packets[0]
        try:
            index = np.argwhere(x_packets[1::2]-x_packets[::2]>0)
        except ValueError:
            index = np.argwhere(x_packets[1::2]-np.flip(x_packets[-3::-2])>0)
        index  = np.concatenate((index*2,(index)*2+1),axis=1).flatten() #returns sorted list
        index = np.split(index,np.where(np.diff(index)!=1)[0]+1)
        packet_times = np.delete(packet_times,index[-1])
        packets = list(map(lambda packet:np.delete(packet,index[-1]),packets))
        try:
            if index[0][0] == 0:
                packet_times = np.delete(packet_times,index[0])
                packets = list(map(lambda packet:np.delete(packet,index[0]),packets))
        except IndexError:
            pass
        if len(packet_times)>3:
            self.regression_functions.fit(packet_times,packets[0])
            index = outliers(
                packets[0]-list(map(self.regression_functions.Predict,packet_times)),
                STANDARD_DEVIATIONS=STANDARD_DEVIATIONS)
            if index.any():
                packet_times = np.delete(packet_times,index)
                packets = list(map(lambda packet:np.delete(packet,index),packets))
        return [packet_times,packets]

    def r_neuron(self,clean_package,r_THRESHOLD=-0.7):
        """requires cleanned package, r_THRESHOLD=rejection threshold for regression
        constant"""
        xaxis,packets = clean_package
        yaxis = packets[0]
        if len(xaxis)>3:
            _,regression_constant = self.regression_functions.fit(xaxis,yaxis)
            if regression_constant < r_THRESHOLD:
                return True
        return None

    def t_neuron(self,package,STANDARD_DEVIATIONS=2.0,MIN_WORDS=4):
        """determines whether enough time has elapsed to have read line,
        STANDARD_DEVIATIONS: standard deviations for estimate reading time,
        MIN_WORDS : min words for line to be auto scrolled"""
        packet_times,_ = package
        self.reading_amount += (packet_times[-1]-packet_times[0])
        try:
            word_count = len(self.extract_functions.page[self.line_index])
        except IndexError:
            self.extract_functions.load_next_page()
            self.line_index = 0
            word_count = len(self.extract_functions.page[self.line_index])
        minimum_reading_time = word_count/(self.extract_functions.reading_rates[-1]
                                   +STANDARD_DEVIATIONS*self.extract_functions.reading_std)
        if self.reading_amount > minimum_reading_time:
            self.reading_amount -= minimum_reading_time
            self.line_index += 1
            try:
                word_count = len(self.extract_functions.page[self.line_index])
            except IndexError:
                self.extract_functions.load_next_page()
                self.line_index = 0
                word_count = len(self.extract_functions.page[self.line_index])
            if word_count < MIN_WORDS:
                #automatically scrolling the line
                minimum_reading_time = word_count/(self.extract_functions.reading_rates[-1]
                                           +STANDARD_DEVIATIONS*self.extract_functions.reading_std)
                self.line_index += 1
                self.reading_amount -= minimum_reading_time
            return True
        return False
"""
for simply loading PDF into useable pythonic form
uses the slower pdfplumber as opposed to pdftotext because it is easier to install,
with pdftotext requiring Anaconda or c++ tools - not a given
"""
class ExtractFunctions:
    """loading PDF file into pythonic form"""
    def __init__(self):
        self.extract = None
        self.page = None
        self.page_index = None
        self.load_reading_rates()
    def load_extract(self,datapath,call=False):
        """loads specified PDF"""
        self.extract = datapath
        if call:
            self.call_extract()

    def call_extract(self):
        """will open PDF in preffered reader"""
        with open("openExtract.bat","w") as file:
            file.write('start "" /max "'+self.extract)
        subprocess.call([r"openExtract.bat"])

    def load_page(self,page_number):
        """will load page for self.extract"""
        self.page_index = page_number
        with pdfplumber.open(self.extract) as file:
            self.page = file.pages[self.page_index]
            self.page = list(map(lambda elem:elem.split("\t"),
                                 self.page.extract_text().split("\n")))
        return self.page

    def load_next_page(self):
        """will load page"""
        self.page_index += 1
        with pdfplumber.open(self.extract) as file:
            self.page = file.pages[self.page_index]
            self.page = list(map(lambda elem:elem.split("\t"),
                                 self.page.extract_text().split("\n")))
        return self.page

    def load_reading_rates(self,file_path=package_path):
        """defines self.reading_rates and self.reading_std attributes"""
        with open(file_path+"reading_rates.txt","r") as file:
            self.reading_rates = np.array(list(map(float,file.read().split("|")[1:])))
        self.reading_rates = np.sort(self.reading_rates,axis=0)
        self.reading_std = self.reading_rates.std()
        return self.reading_rates

"""
simple regression functions using numpy
will calculate gradient,y-intercept and a regression corralation coefficient 
recycles certain variables for efficiency as functions could be called and used often
"""

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
"""
prototype.
Will require user to supply and source their own PDF file
will save the packages by default for analysis however will only involve the dry run tool 
other analysis tools aren't built to a great degree of userability, need to be upgraded - 
potentially to be added to later versions
"""
class prototype:
    """running the current prototype"""
    def __init__(self):
        self.pupil_functions = PupilFunctions()
        self.left_packet_model = packet_model()
        self.right_packet_model= packet_model()

        self.directory = None

    def dry_run(self,save_dir="packageSamples",file_path=package_path):
        """will dry run through saved data to ensure model is working"""
        extract_index = load_extract_index(save_dir=save_dir,file_path=file_path)
        datapath = os.path.join("data",save_dir,"")
        _,directories,_ = next(os.walk(datapath))
        directories = sorted(directories,key=lambda elem:int(elem[7:]))
        for directory_number,directory in enumerate(directories):
            extract,page_number = extract_index[directory]
            self.left_packet_model.load_extract(extract,page_number,call=False)
            self.right_packet_model.load_extract(extract,page_number,call=False)
            left_data = load_data(directory_number,0,
                                  save_dir=save_dir,file_path=file_path)
            left_data_iter = list(zip(*left_data))
            right_data = load_data(directory_number,1,
                                   save_dir=save_dir,file_path=file_path)
            right_data_iter = list(zip(*right_data))
            for index_number in range(len(sorted([left_data_iter,right_data_iter],
                                     key=len)[0])):
                packet_time,packet = left_data_iter[index_number]
                left_package = self.left_packet_model.build_package(packet_time,packet)
                if left_package:
                    pass
                packet_time,packet = right_data_iter[index_number]
                right_package = self.right_packet_model.build_package(packet_time,packet)
                if right_package:
                    pass

    def run(self,extract,page_number,MIN_LINES=3,
            save_dir="packageSamples",file_path=package_path,call=False):
        """running the prototype - for 150% zoom:
        parameters: MIN_LINES=minimum amount of lines read before scrolling,
        save_dir=directory to save the sample to, file_path = where directory should be found,
        call=True/False whether to call PDF doc to PDF reader"""
        #load extracts for both models
        self.left_packet_model.load_extract(extract,page_number,call=call)
        self.right_packet_model.load_extract(extract,page_number,call=False)
        line_count = 0
        start_time = time.perf_counter()
        complete_left = [[],[]]
        complete_right= [[],[]]
        print("preparing webcam, please wait ... ")
        cap = cv2.VideoCapture(0)
        print("ready, <press space>")
        while True:
            if keyboard.is_pressed("space"):
                break
        while True:
            _,frame = cap.read()
            left_pupil,right_pupil = self.pupil_functions.find_pupils(frame)
            if left_pupil:
                complete_left[0].append(time.perf_counter()-start_time)
                complete_left[1].append(left_pupil)
                left_package = self.left_packet_model.build_package(time.perf_counter()-start_time,
                                                                    left_pupil)
                if left_package:
                    line_change=self.left_packet_model.line_index-self.right_packet_model.line_index
                    if line_change < 0:
                        line_count += 1
                    else:
                        line_count += line_change
                        #will scroll a base three lines at a time
                    if line_count >= MIN_LINES:
                        scroll_line(nlines=MIN_LINES)
                        line_count = 0
                    if self.left_packet_model.extract_functions.page_index != self.right_packet_model.extract_functions.page_index:
                        scroll_page()
                        line_count = 0
                        self.left_packet_model.reading_amount = 0
                        self.right_packet_model.reading_amount = 0
                    #equatting the attributes of left and right model
                    self.right_packet_model.extract_functions.page_index = self.left_packet_model.extract_functions.page_index
                    self.right_packet_model.package = [[],[]]
                    self.right_packet_model.line_index = self.left_packet_model.line_index
                    self.right_packet_model.reading_amount = self.left_packet_model.reading_amount

            if right_pupil:
                complete_right[0].append(time.perf_counter()-start_time)
                complete_right[1].append(right_pupil)
                right_package=self.right_packet_model.build_package(time.perf_counter()-start_time,
                                                                    right_pupil)
                if right_package:
                    line_change=self.right_packet_model.line_index-self.left_packet_model.line_index
                    if line_change < 0:
                        line_count += 1
                    else:
                        line_count += line_change
                    if line_count >= MIN_LINES:
                        scroll_line(nlines=MIN_LINES)
                        line_count = 0
                    if self.right_packet_model.extract_functions.page_index != self.left_packet_model.extract_functions.page_index:
                        scroll_page()
                        line_count = 0
                        self.right_packet_model.reading_amount = 0
                        self.left_packet_model.reading_amount = 0
                    self.left_packet_model.extract_functions.page_index = self.right_packet_model.extract_functions.page_index
                    self.left_packet_model.package = [[],[]]
                    self.left_packet_model.line_index = self.right_packet_model.line_index
                    self.left_packet_model.reading_amount = self.right_packet_model.reading_amount
            if keyboard.is_pressed("space"):
                break
            cv2.waitKey(1)
            cv2.imshow("frame",frame)
        cap.release()
        cv2.destroyAllWindows()
        create_data_directory(save_dir=save_dir,file_path=file_path)
        self.save_package(complete_left,save_dir=save_dir,file_path=file_path)
        self.save_package(complete_right,save_dir=save_dir,file_path=file_path)

    def save_package(self,package,save_dir="packageSamples",file_path=package_path):
        """saving a prototype run for analysis"""
        #checking if a directory has been made to save to
        if not self.directory:
            datapath = os.path.join("data",save_dir,"")
            _,directories,_ = next(os.walk(file_path+datapath))
            try:
                self.directory = "sample_"+str(max(list(map(lambda elem:int(elem[7:]),
                                                            directories)))+1)
            except ValueError:
                self.directory = "sample_0"
            datapath = os.path.join("data",save_dir,self.directory)
            os.mkdir(file_path+datapath)
        #deciding what the name of the saved package will be
        datapath = os.path.join("data",save_dir,self.directory,"")
        _,_,packages = next(os.walk(file_path+datapath))
        try:
            package_route = "package_"+str(max(list(map(lambda elem:int(elem[8:-4]),
                                                       packages)))+1)+".txt"
        except ValueError:
            package_route = "package_0.txt"
        #formatting package to be saved as a .txt file
        times,packets = package
        packets = "-".join(list(map(lambda elem:",".join([str(i) for i in elem]),packets)))
        times   = "-".join([str(i) for i in times])
        package_string = times+"|"+packets
        with open(file_path+datapath+package_route,"w") as file:
            file.write(package_string)
