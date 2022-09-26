# -*- coding: utf-8 -*-

import os
import time
# from string import ascii_letters,digits
# import random
# import csv
import cv2
import pyautogui
import keyboard
import psutil
import tkinter
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
            self.pupils[indx] =(pupil_x+face_x+eye_x, face_y+eye_y-pupil_y)
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
    difference = np.split(index,np.where(difference>6)[0]+1)
    # if 'gap' between black pixels is enough will seperate a word
    if len(difference) > 0:
        words = [line[:,dif[0]-1:dif[-1]+1] for dif in difference if len(dif) > 0]
    line_size = index.max() - index.min() if len(index) else 0
    return words, line_size

def determine_browser():
    """chrome or internet_explorer, uses psutil"""
    running_apps = list(map(lambda app:app.name(),psutil.process_iter()))
    #chrome functions work with edge but not vice versa
    if "chrome.exe" in running_apps:
        return "chrome.exe" #most used browser
    return "MicrosoftEdge.exe" # default edge

class ScreenFunctionsEExplorer:
    def __init__(self):
        """analysing screen when using Enternet Explorer to view the .pdf"""
        self.frame = None
        self.fframe = None
        self.lines = np.array([])
        self.screen_height,self.screen_width = np.array(pyautogui.screenshot()).shape[:2]

    def set_frame(self):
        """take screenshot, and convert to cv2 format"""
        self.frame = np.array(pyautogui.screenshot())
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        return self.frame

    def filter_frame(self):
        """cut and filter frame""" #binarise and make easier to seperate
        #fframe = filtered frame, who cares about naming conventions?
        # get the appropriate area of the screen
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
            return [(len(count),(width,height)) for (count,width),height in zip(word_count,index)]
        # return word_count,(shape of line)
        return None

def cut_frame_chrome(frame):
    """cut std chrome frame to size"""
    #remove top
    frame_height = frame.shape[0]
    frame_sum = frame.sum(axis=1)
    frame_sum_min = frame_sum.min()
    index1 = np.argwhere(frame_sum[:frame_height//4]==frame_sum_min)
    index1 = [[0]] if not len(index1) else index1
    frame = frame[index1[-1][-1]+1:]
    #remove sides
    frame_sum = frame.sum(axis=0)
    frame_sum_min = frame_sum.min()
    index1 = np.argwhere(frame_sum[:frame_height//2]==frame_sum_min) #left
    index1 = [[0]] if not len(index1) else index1
    index2 = np.argwhere(frame_sum[frame_height//2:]==frame_sum_min) #right
    index2 = [[-1]] if not len(index2) else index2 + frame.shape[0]//2
    frame = frame[:,index1[-1][-1]:index2[0][0]]
    return frame

class ScreenFunctionsChrome:
    def __init__(self):
        """analysing screen when using google Chrome to view the .pdf"""
        self.frame = None
        self.fframe = None
        self.lines = np.array([])
        self.screen_height,self.screen_width = np.array(pyautogui.screenshot()).shape[:2]

    def set_frame(self):
        """take screenshot, and convert to cv2 format"""
        self.frame = np.array(pyautogui.screenshot())
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        return self.frame

    def filter_frame(self):
        """cut and filter frame"""
        self.fframe = self.frame[int(self.screen_height*0.1):int(self.screen_height*0.9),
                                 int(self.screen_width*0.05):-int(self.screen_width*0.05)]
        _,self.fframe = cv2.threshold(self.fframe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.fframe = cut_frame_chrome(self.fframe) #cut large sections of black a result of chrome pdf reader format
        self.fframe = cv2.inRange(self.fframe,0,254)
        return self.fframe
    #code following same as ScreenFunctionsEExplorer:

    def seperate_lines(self, whitespace=False):
        """seperate frame into lines of writing"""
        index = np.argwhere(self.fframe>0)
        index = np.unique(index[:,0])
        difference = index[1:]-index[:-1]
        if not whitespace:
            difference = np.split(index,np.where(difference>1)[0]+1)
        else:
            difference = np.split(index,np.where(difference>1)[0])
        if len(difference) > 1:
            self.lines = np.array([self.fframe[dif[0]-1:dif[-1]+1] for dif in difference
                                   if len(dif) > 0],dtype=object)
            if whitespace:
                self.lines[0] = np.concatenate((self.fframe[0:index[0]],
                                                self.lines[0]))
        return self.lines

    def analyse_lines(self):
        """return height and word count of each line"""
        self.set_frame()
        self.filter_frame()
        self.seperate_lines(whitespace=True)
        if len(self.lines) > 0:
            index = np.array([line.shape[0] for line in self.lines][:-1])
            lines = self.lines[np.argwhere(index>np.floor(index.mean()-index.std()*1.5))[:,0]]
            word_count = list(map(seperate_words,lines))
            index = list(map(lambda elem:elem.shape[0],lines))
            return [(len(count),(width,height)) for (count,width),height in zip(word_count,index)]
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

class PackageModel:
    def __init__(self):
        """new package algorithm"""
        self.package = [[],[]]
        self.time_read = 0

        self.reading_rate = 6.00 # average reading rate of 180 - 320 wpm (for fiction), using upper # ref Google
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

class Settings:
    def __init__(self):
        """seperate window for settings"""
        self.tk = None
        self.frame = None

        self.reading_rate = 6.0
        self.minimum_lines = 2
        self.scroll_multiplier = 1.2
        self.display = True

        self.load_settings()

    def load_settings(self,file_path=package_path):
        """load prior used settings"""
        datapath = os.path.join(file_path,"settings.txt")
        if not os.path.exists(datapath):
            return None
        with open(datapath,"r") as file:
            self.reading_rate,self.minimum_lines,self.display = file.readlines()
        self.reading_rate = float(self.reading_rate[:-1]) #convert to useable form
        self.minimum_lines = int(self.minimum_lines[:-1])
        self.display = bool(int(self.display))
        return self.reading_rate,self.minimum_lines,self.display

    def save_settings(self,file_path=package_path):
        """save settings for next use"""
        datapath = os.path.join(file_path,"settings.txt")
        towrite = [self.reading_rate,self.minimum_lines,int(self.display)] #to write to file
        towrite = "\n".join(list(map(str,towrite))) # change to writeable form
        with open(datapath,"w") as file:
            file.writelines(towrite)
        return file_path

    def open_window(self,file_path=package_path):
        """create settings window, fixed height, width"""
        self.tk = tkinter.Tk()
        self.tk.title("settings")
        self.tk.iconbitmap(os.path.join(file_path,"icon.ico"))

        self.tk.minsize(205,215)
        self.tk.maxsize(205,215)

        tkinter.Label(self.tk,text="Reading Rate (Words Per Minute):").grid()
        def set_reading_rate(event):
            self.reading_rate = int(event)/60
        wpm_slider = tkinter.Scale(self.tk,from_=50,to=600,resolution=10,orient="horizontal",
                                    command=set_reading_rate,length=200)
        wpm_slider.set(self.reading_rate*60)
        wpm_slider.grid(pady=1)

        def set_display():
            self.display = not self.display
        display =tkinter.Checkbutton(self.tk,text="Show Webcam",command=set_display)
        if self.display:
            display.toggle()
        display.grid(pady=10)

        tkinter.Label(self.tk,text="Min Lines Before Scroll:").grid()
        def set_minimum_lines(event):
            self.minimum_lines = int(event)
        min_lines_slider = tkinter.Scale(self.tk,from_=2,to=10,resolution=1,orient="horizontal",
                                         command=set_minimum_lines,length=200)
        min_lines_slider.set(self.minimum_lines)
        min_lines_slider.grid()

        def reset_defaults():
            if not self.display:
                display.toggle()
            self.display = True
            self.reading_rate = 6.0
            wpm_slider.set(360)
            self.minimum_lines = 2
            min_lines_slider.set(2)

        reset_button = tkinter.Button(self.tk,text="Reset To Defaults",command=reset_defaults)
        reset_button.grid(pady=10)

class gui:
    def __init__(self,file_path=package_path):
        """gui for easy starting of program, contains 1 button"""
        self.tk = tkinter.Tk()
        self.tk.title("scrollthescroll")
        self.tk.iconbitmap(os.path.join(file_path,"icon.ico"))

        self.button = None #button intended to be changeable
        self.settings = Settings() #settings will be another button, not destroyable or changeable

    def create_window(self):
        """create gui window, using fixed height and width"""
        #size and create basic frame:
        self.tk.minsize(250,375)
        self.tk.maxsize(250,375)
        #add blank button for loading webcam
        self.button = tkinter.Button(self.tk,height=2,width=32,
                                     text="button")
        self.button.grid(row=0,padx=8,pady=8)
        settings_button = tkinter.Button(self.tk,height=1,width=16,text="settings",
                                         command=self.settings.open_window)
        settings_button.grid(row=1,padx=8,pady=285)
        return self.tk

    def bind_button(self,command=None,text=None,keypress=None):
        """change the button"""
        self.button.destroy()
        self.button = tkinter.Button(self.tk,height=2,width=32,
                                     text=text,command=command)
        self.button.grid(row=0,padx=8,pady=8)
        if keypress:
            self.tk.bind(keypress,command)
        return self.button

class Prototype:
    def __init__(self):
        """new prototype utilizes GUI"""
        self.right_packet_model = PackageModel()
        self.left_packet_model = PackageModel()

        self.pupil_functions = PupilFunctions()
        self.screen_functions = None

        self.screen_analysis = None
        self.scroll_lines = 0

        self.gui = None
        self.cap = None

    def load_screen_functions(self):
        """finding the browser used"""
        browser = determine_browser()
        if browser == "chrome.exe":
            self.screen_functions = ScreenFunctionsChrome()
            return True
        if browser == "MicrosoftEdge.exe":
            self.screen_functions = ScreenFunctionsEExplorer()
            return True
        return False

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

    def run_gui(self):
        """starts gui, awaits command to prepare VideoCapture"""
        self.gui = gui()
        self.gui.create_window()

        def button_function(event=None):
            self.gui.bind_button(command=None,text="loading webcam, please wait ...")
            self.gui.tk.after(1000,load_video_capture)

        def run_algorithm(event=None,file_path=package_path):
            self.left_packet_model.reading_rate = self.gui.settings.reading_rate
            self.right_packet_model.reading_rate = self.gui.settings.reading_rate
            self.run_algorithm(file_path=file_path,minimum_lines=self.gui.settings.minimum_lines,
                               scroll_multiplier=self.gui.settings.scroll_multiplier,
                               display=self.gui.settings.display)

        def load_video_capture(): #load self.cap, takes a long time
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.gui.bind_button(command=None,text="error, cannot open webcam")
                raise IOError("error, cannot open webcam")
            self.gui.bind_button(command=run_algorithm,text="ready, <press space>",keypress="<space>")

        self.gui.bind_button(command=button_function,text="press space to begin",keypress="<space>")
        self.gui.tk.mainloop()

    def run_algorithm(self,file_path=package_path,minimum_lines=2,scroll_multiplier=1.2,display=True):
        """runs the scrollthescroll algorithm"""
        self.gui.tk.destroy()
        try:
            self.gui.settings.tk.destroy()
        except:
            pass
        self.gui.settings.save_settings(file_path=file_path)
        self.pupil_functions.load_haars(file_path=file_path)
        read_lines = 0
        analysis_time = start_time = time.perf_counter()
        while True:
            if time.perf_counter() - analysis_time > 0.5:
                if not self.screen_functions:
                    if not self.load_screen_functions():
                        analysis_time = time.perf_counter()
                        continue
                analysis = self.screen_functions.analyse_lines()
                if analysis != self.screen_analysis:
                    if self.load_line_analysis():
                        analysis_time = time.perf_counter()
                    else:
                        continue
                else:
                    analysis_time = time.perf_counter()
            _,frame = self.cap.read()
            left_pupil,right_pupil = self.pupil_functions.find_pupils(frame)
            if left_pupil:
                left_package = self.left_packet_model.build_package(time.perf_counter()-start_time,
                                                                    left_pupil)
                if left_package:
                    read_lines += 1
                    self.scroll_lines += 1
                    if read_lines >= minimum_lines:
                        self.scroll(scroll_multiplier=scroll_multiplier)
                        self.screen_analysis = self.load_line_analysis()
                        analysis_time = time.perf_counter()
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
                        self.scroll(scroll_multiplier=scroll_multiplier)
                        self.screen_analysis = self.load_line_analysis()
                        analysis_time = time.perf_counter()
                        read_lines = 0
                    self.right_packet_model.package = [[],[]]
                    self.left_packet_model.package = [[],[]]
                    self.left_packet_model.time_read = self.right_packet_model.time_read
            if display:
                cv2.imshow("frame",frame)
            cv2.waitKey(1)
            if keyboard.is_pressed("esc"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = Prototype()
    p.run_gui()
