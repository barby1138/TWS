import os
from os import listdir
from os.path import isfile, join
import shutil
import traceback

import sys

from tkinter import filedialog
from tkinter import *
import tkinter as tk

import pprint

import numpy as np
from PIL import Image, ImageOps
from PIL import ImageTk

import cv2
import json

from pathlib import Path

import blender

import matplotlib.pyplot as plt

import math
import random

#TODO config
SAMPLE_SROPOSALS_MIN = 2
SAMPLE_SROPOSALS_MAX = 10
ANGLE_DELTA = 60

def TextEdit(main_root, text):
    top = tk.Toplevel(main_root)
    main_text = tk.Text(top)
    ret = []

    def accept():
        ret.append(main_text.get("1.0", END))
        top.quit()
        pass

    def cancel():
        ret.append(text)
        top.quit()
        pass

    btOk = tk.Button(top, height=1, width=10, text="OK", command=accept)
    btNo = tk.Button(top, height=1, width=10, text="Cancel", command=cancel)
    btNo.pack()
    btOk.pack()

    main_text.insert(END, text)
    main_text.pack()

    top.update()

    top.mainloop()
    top.destroy()

    return ret

class PannelFg_obj(tk.Label):
    def __init__(self, master, parent, image, file):
        super().__init__(master=master, image=image)
        self.file = file
        self.parent = parent

    def on_select(self, event):
        print(self.file)
        self.parent.parent.update_fg(self.file)

class PannelFg:

    def __init__(self, master, parent, objs_path):
        self.parent = parent

        top = tk.Toplevel(master)

        top.winfo_toplevel().title('FG')
        def cancelCommand(event=None): pass
        top.protocol( 'WM_DELETE_WINDOW', cancelCommand )

        top.bind("<Key>", self.on_press_key)

        onlyfiles = [os.path.join(objs_path, f) for f in listdir(objs_path) if
                        isfile(join(objs_path, f)) and f.endswith(".png")]

        #print(onlyfiles)

        hscrollbar = tk.Scrollbar(top)

        self.imgs = []

        c = 0
        column = 0
        row = 0
        ROW_MAX = 9 # 0..9 == 10 rows
        #TODO 
        # read json
        # search by class / angle / light / etc.
        # scroll (left - right)
        for file in onlyfiles:
            image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.thumbnail([100,100],Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)

            lb = PannelFg_obj(top, self, image=image, file=file)
            lb.grid(row=row, column=column, sticky='w')

            if row >= ROW_MAX:
                column += 1
                row = 0
            else:
                row += 1

            lb.bind("<Button-1>", lb.on_select)
            self.imgs.append(image)

        #self.parent.sel_fg_path = onlyfiles[0]

    def on_press_key(self, event):
        c = event.keysym
        s = event.state

        ctrl = (s & 0x4) != 0
        #print(ctrl)
        #print(s)
        #print(c)
    
        if c == "Left":
            print("left")
        
        if c == "Right":
            print("right")

class PannelBV_obj(tk.Label):
    def __init__(self, master, parent, image, file):
        super().__init__(master=master, image=image)
        self.file = file
        self.parent = parent

    def on_select(self, event):
        print(self.file)
        self.parent.parent.update_fg(self.file)

class PannelBV:

    def __init__(self, master, parent):
        self.parent = parent

        top = tk.Toplevel(master)

        top.winfo_toplevel().title('BV')
        def cancelCommand(event=None): pass
        top.protocol( 'WM_DELETE_WINDOW', cancelCommand )

        top.bind("<Key>", self.on_press_key)

        self.top = top

        hscrollbar = tk.Scrollbar(top)

        self.imgs = []

        self.delta = 0

        file = parent.img_bg_path
        print(file)

        img = cv2.imread(file)
        image= Image.fromarray(img)
        image = ImageTk.PhotoImage(image)
        
        self.canvas = Canvas(self.top,  width=image.width(), 
                                        height=image.height())

        self.render(file)
        
        parent.register_nav_cb(self.on_nav)
    
        self.curr_points = []
        self.inv_curr_points = []
        self.traj_arr = []

        frm = self.canvas
        frm.bind("<Button-1>", self.on_press)
        frm.bind("<Button-2>", self.on_press)
        frm.bind("<Button-3>", self.on_press)
        frm.bind("<Button-4>", self.on_mousewheel)
        frm.bind("<Button-5>", self.on_mousewheel)

        # TODO calculate
        self.car_h = 30
        self.car_w = 20

        self.not_intersect=True
        self.do_sample_proposals=True
        self.sample_proposals_num=5
        self.do_render_fg=False
        self.proposals = [] # sampled
        self.all_proposals = []
        self.idx = []

    def render(self, file):
        #img = cv2.imread(file)
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        image = ImageTk.PhotoImage(image)
        self.img_bg = image

        delta = self.delta
        IMAGE_H = delta + self.img_bg.height() // 2 #223
        IMAGE_W = self.img_bg.width() #1280

        delta_X = 569

        src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[delta_X, IMAGE_H], [IMAGE_W - delta_X, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        #print(src)
        #print(dst)
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        self.Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

        img = img[(self.img_bg.height() - IMAGE_H):self.img_bg.height(), 0:IMAGE_W] # Apply np slicing for ROI crop
        #print(img)
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
        #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
        #plt.show()
        image = Image.fromarray(warped_img)
        image = ImageTk.PhotoImage(image)
        
        self.img=image
        self.img_bg_tag = self.canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.canvas.pack(fill=BOTH, expand=True)
    
        delta_T = (IMAGE_H*IMAGE_W)/(2*delta_X) - IMAGE_H
        #print(delta_T)

        # TODO find distance to visor

        self.parent.set_M(M)
        self.parent.set_delta_T(delta_T)
        
    def on_nav(self, event):
        #print(event)
        file = self.parent.img_bg_path
        #print(file)

        c = event.keysym
        s = event.state
        
        if c == "Up":
            self.delta += 1
            #print(self.delta)
            #self.render(file)
        
        if c == "Down":
            self.delta -= 1
            #print(self.delta)
            #self.render(file)

        self.render(file)

    def on_press_key(self, event):
        c = event.keysym
        s = event.state

        ctrl = (s & 0x4) != 0
        #print(ctrl)
        #print(s)
        print(c)

        file = self.parent.img_bg_path

        if c == "Up":
            self.delta += 1
            print(self.delta)
            self.render(file)

        if c == "Down":
            self.delta -= 1
            print(self.delta)
            self.render(file)

        if c == "i":
            print("togle/utogle all / intersect proposals")
            self.not_intersect = not self.not_intersect

            self.render(file)
            self.render_proposals()

        if c == "r":
            print("togle/utogle render fg")
            self.do_render_fg = not self.do_render_fg


        if c == "plus":
            print("inc sample proposals num")
            if self.sample_proposals_num < SAMPLE_SROPOSALS_MAX:
                self.sample_proposals_num = self.sample_proposals_num + 1

        if c == "minus":
            print("dec sample proposals num")
            if self.sample_proposals_num > SAMPLE_SROPOSALS_MIN:
                self.sample_proposals_num = self.sample_proposals_num - 1

        if c == "a":
            print("togle/utogle all intersect / sample proposals")
            self.do_sample_proposals = not self.do_sample_proposals

            self.render(file)
            self.render_proposals()

        if c == "s":
            print("new sample proposals")

            self.do_sample_proposals = True
            self.not_intersect = True

            self.reset_idx()
            self.render(file)
            self.render_proposals()

        if c == "n":
            print("new intersect proposals")

            self.do_sample_proposals = False
            self.not_intersect = True

            self.generate_proposals()
            self.render(file)
            self.render_proposals()

    def on_press(self, event):
        #print(event)
        if event.type in [EventType.ButtonPress] and event.num == 1:
            #print('line')
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)

            points = np.array([[x, y]], dtype='float32')
            points = np.array([points])

            inv_point = cv2.perspectiveTransform(points, self.Minv)
            #print(inv_point[0][0])
            self.inv_curr_points.append([inv_point[0][0][0], inv_point[0][0][1] + self.delta + self.img_bg.height() // 2])

            self.curr_points.append([x,y])
            flat_list = [item for points in self.curr_points for item in points]
            #print(flat_list)
            if len(self.curr_points) > 1:
                self.canvas.create_line(*flat_list, fill="green", width=5, arrow=tk.LAST)

        elif event.type in [EventType.ButtonPress] and event.num == 3:
            if len(self.curr_points) > 1:
                self.traj_arr.append(self.curr_points)
                self.generate_proposals()
                file = self.parent.img_bg_path
                self.render(file)
                self.render_proposals() #self.curr_points)

            self.curr_points = []
            self.inv_curr_points = []

    def get_actual_proposals(self):
        if self.not_intersect:
            return self.sample_proposals() 
        else:
            return self.all_proposals

    def render_proposals(self):
        #self.generate_proposals(traj_arr)
        self.draw_proposals(self.get_actual_proposals())

        self.parent.draw_proposals(self.inv_proposals(self.get_actual_proposals()), self.do_render_fg)

        for traj_points in self.traj_arr:
            self.draw_trajectory(traj_points)

    def reset_idx(self):
        if (self.sample_proposals_num < len(self.proposals)) and self.do_sample_proposals:
            self.idx = random.sample(range(len(self.proposals)), self.sample_proposals_num)
        else:
            print("no need to reset_idx")
            #self.idx = range(len(self.proposals))

    def sample_proposals(self):
        if (self.sample_proposals_num < len(self.proposals)) and self.do_sample_proposals:
            #if len(self.idx) == 0:
             #   print("set idx")
              #  self.idx = random.sample(range(len(self.proposals)), self.sample_proposals_num)
            idx = self.idx #random.sample(range(len(self.proposals)), self.sample_proposals_num)
            proposals = [a for i, a in enumerate(self.proposals) if i in idx]
            return  proposals
        else:
            print("no need to sample")
            return self.proposals

    def generate_proposals(self):
        proposals = []
        all_proposals = []
        for traj_points in self.traj_arr:
            traj_proposals = self.traj_to_proposals(traj_points)
            all_proposals = all_proposals + traj_proposals

            if self.not_intersect:
                s_traj_proposals = self.non_intersect_proposals(traj_proposals)
            
                #print(len(proposals))
                for proposal in proposals:
                    for i, s_traj_proposal in enumerate(s_traj_proposals):
                        if s_traj_proposal is None:
                            continue 
                        if self.do_proposals_intersect(proposal, s_traj_proposal):
                            #print("intersect - set None")
                            s_traj_proposals[i] = None
                            #proposal = None
                        
                result = [a for a in s_traj_proposals if a is not None]
                
                proposals = proposals + result
                #print(len(proposals))
            else:
                proposals = proposals + traj_proposals


        self.proposals = self.sort_proposals(proposals)
        self.all_proposals = self.sort_proposals(all_proposals)
        self.reset_idx()

    def sort_proposals(self, proposals):
        proposals_t = [tuple(p) for p in proposals]
        proposals_np = np.array(proposals_t, dtype=[('x', float), ('y', float), ('a_0', float), ('a_1', float), ('d', float)])
        proposals_np = np.sort(proposals_np, order='y')
        return proposals_np.tolist()

    def inv_proposals(self, proposals):
        inv_proposals = []
        for point in proposals:
            print(point)
            x = point[0]
            y = point[1]
            points = np.array([[x, y]], dtype='float32')
            points = np.array([points])

            inv_point = cv2.perspectiveTransform(points, self.Minv)
            print(inv_point[0][0])
            inv_proposals.append([inv_point[0][0][0], 
                                    inv_point[0][0][1] + self.delta + self.img_bg.height() // 2, 
                                    point[2], 
                                    point[3], 
                                    point[4]])

        return inv_proposals

    def draw_trajectory(self, traj_points):
        #print("draw traj")
        flat_list = [item for points in traj_points for item in points]
        #print(flat_list)
        if len(traj_points) > 1:
            self.canvas.create_line(*flat_list, fill="green", width=5, arrow=tk.LAST)

    def draw_proposals(self, proposals):
        for prop in proposals:
            # prop[3]
            points = self.rotated_rect([prop[0], prop[1]], (prop[2]), self.car_w, self.car_h)
            #self.canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="red")
            self.canvas.create_polygon(points, outline='#f11',fill='#1f1', width=2)

    def do_proposals_intersect(self, prop1, prop2):
        points_1 = self.rotated_rect([prop1[0], prop1[1]], (prop1[2]), self.car_w, self.car_h)
        rect_1 = cv2.minAreaRect(np.array(points_1).astype(np.int32))

        points_2 = self.rotated_rect([prop2[0], prop2[1]], (prop2[2]), self.car_w, self.car_h)
        rect_2 = cv2.minAreaRect(np.array(points_2).astype(np.int32))

        int = cv2.rotatedRectangleIntersection(rect_1, rect_2)

        return (int[0] > 0)
            
    def non_intersect_proposals(self, proposals):
        
        start = random.randrange(len(proposals))

        #fwd 
        for i in range(start, len(proposals)-1, 1):
            if proposals[i] == None:
                continue

            for j in range(i+1, len(proposals), 1):
                if proposals[j] == None:
                    continue

                if self.do_proposals_intersect(proposals[i], proposals[j]):
                    proposals[j] = None

        #bwd 
        for i in range(start, 0, -1):
            if proposals[i] == None:
                continue

            for j in range(i-1, -1, -1):
                if proposals[j] == None:
                    continue

                if self.do_proposals_intersect(proposals[i], proposals[j]):
                    proposals[j] = None

        result = [a for a in proposals if a is not None]

        return result

    # TODO to util class?
    def rotated_rect(self, centre, theta, width, height):
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        # print(R)
        p1 = [ + width / 2,  + height / 2]
        p2 = [- width / 2,  + height / 2]
        p3 = [ - width / 2, - height / 2]
        p4 = [ + width / 2,  - height / 2]
        p1_new =  np.squeeze(np.asarray(np.dot(p1, R)+ centre)).tolist()
        p2_new =  np.squeeze(np.asarray(np.dot(p2, R)+ centre)).tolist()
        p3_new =  np.squeeze(np.asarray(np.dot(p3, R)+ centre)).tolist()
        p4_new =  np.squeeze(np.asarray(np.dot(p4, R)+ centre)).tolist()
        p_new = [p1_new,p2_new,p3_new,p4_new]
        #print(p_new)
        return p_new

    def traj_to_proposals(self, traj_points):
        #print("traj_to_proposals")
        dL = 10
        pairs = []
        proposals = []
        traj_points_len = len(traj_points)
        #print(traj_points_len)
        if traj_points_len > 1:
            for i in range(traj_points_len-1):
                #print(i)
                pair = [traj_points[i], traj_points[i+1]]
                #print(pair)
                pairs.append(pair)

            alfa = 0
            for pair in pairs:
                dx = pair[1][0] - pair[0][0]
                dy = pair[1][1] - pair[0][1]
                
                #find alfa
                if dy == 0:
                    alfa = np.pi / 2
                else:
                    alfa = np.arctan(dx/dy)

                # to == 1 from == -1
                if dy > 0:
                    direction = 1
                else:
                    direction = -1

                #find alfa_v
                # TODO obtain
                visor_dist = 5
                h = self.img.height()
                w = self.img.width()

                #alfa_v = 
                L = np.sqrt(dx**2 + dy**2)
                #print(L)
                L_slices = math.ceil(L/dL) -1
                #print(L_slices)
                ddx = dL * dx / L
                ddy = dL * dy / L
                #print(ddx)
                #print(ddy)
                for i in range(0, L_slices):
                    p_x = i*ddx + pair[0][0]
                    p_y = i*ddy + pair[0][1]
                    if ((h + visor_dist) - p_y) == 0:
                        alfa_v = np.pi / 2
                    else:
                        alfa_v = np.arctan((p_x - w//2) / ((h + visor_dist) - p_y))

                    new_point = [p_x, p_y, alfa * 180 / 3.14, alfa_v * 180 / 3.14, direction]
                    print(new_point)
                    # start point + new points
                    proposals.append(new_point)

            # .. and last point
            p_x = traj_points[traj_points_len-1][0]
            p_y = traj_points[traj_points_len-1][1]
            if ((h + visor_dist) - p_y) == 0:
                alfa_v = np.pi / 2
            else:
                alfa_v = np.arctan((p_x - w//2) / ((h + visor_dist) - p_y))

            proposals.append([p_x, p_y, alfa * 180 / 3.14, alfa_v * 180 / 3.14, direction])
        
        return proposals

    def on_mousewheel(self, event):
        print(event)

class Fg:

    def __init__(self, parent, x, y, path, scale=1.0, angle=0.0, mirror=False, mode='Load'):
        SIZE_MAX = 300

        # TODO handle FileNotFoundError?
        img = Image.open(path)
        w, h = img.width, img.height
        
        self.test_abs_size = 1.7 #m
        self.scale_abs = h / self.test_abs_size # pix per m
       
        if mode == 'New' and (w > SIZE_MAX or h > SIZE_MAX):
            scale = SIZE_MAX / max(w,h)
            w_o, h_o = int(img.width*scale), int(img.height*scale)
            img_resize = img.resize((w_o, h_o), Image.ANTIALIAS)
        else:
            img_resize = img
        
        img_resize = img

        img_tk = ImageTk.PhotoImage(img_resize)
        img_tag = parent.canvas.create_image(x, y, image=img_tk)
        parent.canvas.tag_bind(img_tag, '<Button-1>', self.on_fg_click)

        self.x = x
        self.y = y

        try:
            with open(path.replace('.png', '.json')) as json_file:  
                print(path.replace('.png', '.json'))
                fg_json = json.load(json_file)
            print('loaded %s' % fg_json)
        except FileNotFoundError:
            # default json
            fg_json={"class_name":"car"}
            print('default %s' % fg_json)

        try:
            self.class_name = fg_json["class_name"]
        except KeyError:
            print("WARNING!!! corrupted json %s" % fg_json)
            self.class_name = 'unk'

        self.b_scaling = False
        if str(path).find('15_6147.png') != -1:
            print("set sc_fg")
            self.b_scaling = True    
            parent.scaling_fg = self

        if str(path).find('_6676.png') != -1:
            print("bu")
            self.scale_abs = self.scale_abs * 0.85

        self.img = img
        # is needed
        self.img_tk = img_tk
        self.img_tag = img_tag
        self.path = path
        self.parent = parent
        self.angle = angle
        self.scale = scale #1.0
        self.mirror = mirror

        self.render_inner()

    def render_inner(self):

        w = math.ceil(self.img.width * self.scale)
        h = math.ceil(self.img.height * self.scale)

        img = self.img.resize((w, h), Image.ANTIALIAS)
        if self.mirror == True:
            img = ImageOps.mirror(img)
        img = img.rotate(self.angle, expand=True)

        self.img_tk = ImageTk.PhotoImage(img)

        self.parent.canvas.itemconfig(self.img_tag, image = self.img_tk)

        if self.b_scaling:
            # for autoscale DEMO
            #print([self.x, self.y])
            self.scale_1_L = self.y + h//2 # bottom of obj
            self.scale_1_H = h / (self.scale_abs*self.test_abs_size)# bottom of obj
            
            #print('scale_1 H-L1: ', self.scale_1_L, 'img_H 1.7m per pix X1: ', self.scale_1_H)
            #self.scale_abs = h / self.test_abs_size


    def render(self, pt=None, delta=None, path=None, act=None):
        if pt is not None:
            deltax = pt[0] - self.x
            deltay = pt[1] - self.y
            self.parent.canvas.move(self.img_tag, deltax, deltay)
            self.x += deltax
            self.y += deltay

            #h = int(self.img.height * self.scale)
            #self.parent.canvas.create_line(self.x, self.y + h//2, self.x, self.y+1+h//2, fill="green", width=10)
            self.render_inner()

        if delta and act is not None:
            if act == 'resize':
                scale = delta * 0.01
                self.scale += scale
                self.render_inner()
            elif act == 'rotate':
                angle = delta
                self.angle += angle
                self.render_inner()

        if act == 'mirror':
            self.mirror = not self.mirror
            self.render_inner()
            
        if path is not None and act == 'update':
            # TODO check if path the same
            self.img = Image.open(path)
            self.path = path
            self.render_inner()

    def on_fg_click(self, event):
        print(event)
        self.parent.curr_fg = self
 
class PanelImg(tk.Frame):

    def shift_zorder(self, inc):
        def shift_inner(a,b):
                self.fg_arr[b], self.fg_arr[a] = self.fg_arr[a], self.fg_arr[b]
                
                start = min(a,b)
                for k in range(start,len(self.fg_arr)):
                    self.canvas.delete(self.fg_arr[k].img_tag)

                for k in range(start,len(self.fg_arr)):
                    self.fg_arr[k].img_tag = \
                        self.canvas.create_image(self.fg_arr[k].x, self.fg_arr[k].y, image=self.fg_arr[k].img_tk)
                    self.canvas.tag_bind(self.fg_arr[k].img_tag, '<Button-1>', self.fg_arr[k].on_fg_click)

        if self.curr_fg is not None:
            a = self.fg_arr.index(self.curr_fg)
            
            if inc > 0 and a < len(self.fg_arr) - 1:
                b = a + 1
                shift_inner(a,b)

            if inc < 0 and a > 0:
                b = a - 1
                shift_inner(a,b)

    def render_fg(self, pt=None, delta=None, path=None, act=None):
        if self.curr_fg is not None:
            self.curr_fg.render(pt=pt, delta=delta, path=path, act=act)

    def delete_fg(self):
        if self.curr_fg is not None:
                if self.curr_fg.b_scaling == False:
                    a = self.fg_arr.index(self.curr_fg)
                    self.canvas.delete(self.fg_arr[a].img_tag)
                    del self.fg_arr[a]
                    self.curr_fg = None
                else:
                    self.canvas.delete(self.curr_fg.img_tag)
                    self.curr_fg = None
                    self.scaling_fg = None
                    

    def update_fg(self, new_path):
        self.canvas.focus_set()
        self.sel_fg_path = new_path
        self.render_fg(act='update', path=new_path)

    def new_fg(self, x, y, path=None):
        #print(self.sel_fg_path)
        if path is None:
            path = self.sel_fg_path

        self.curr_fg = Fg(self, x, y, path, mode='New')
        self.fg_arr.append(self.curr_fg)

    ##################### BV

    def register_nav_cb(self, nav_cb):
        self.nav_cb = nav_cb

    def angle_ann_2_angle(self, angle_ann):
        EPS=1e-8
        angle_ann_arr =  angle_ann.split(":")
        direction = angle_ann_arr[0]
        angle = int(float(angle_ann_arr[1]))
        angle = angle + EPS
        #print(angle)
        if direction == "from" :
            angle = (180 - np.abs(angle))*(angle*(-1)/np.abs(angle))
        # not changed for to
        #print(angle)
        angle = 180 - angle
        #print(angle)
        return np.ceil(angle)

    def angle_is_in_range(self, angle_to_check, delta, angle):
        # delta > 0
        result = False
        ranges = []
        if angle + delta > 360:
            ranges.append([angle, 360])
            ranges.append([0, (angle + delta) - 360])
        else:
            ranges.append([angle, angle + delta])
            
        if angle - delta < 0:
            ranges.append([0, angle])
            ranges.append([360 - (angle - delta), 360])
        else:
            ranges.append([angle - delta, angle])

        for range in ranges:
            if angle_to_check >= range[0] and angle_to_check <= range[1]:
                print("angle match found angle_to_ch %d angle %d range: %d,%d, delta: %d" % (angle_to_check, angle, range[0], range[1], delta))
                result = True
                break

        return result
        pass

    def draw_proposals(self, inv_points, render_fg):
        if render_fg and self.scaling_fg is not None:
            print("render_fg")
            #reset
            [self.canvas.delete(k.img_tag) for k in self.fg_arr if k is not self.scaling_fg]                
            self.fg_arr = [k for k in self.fg_arr if k is self.scaling_fg]

            [self.canvas.delete(i) for i in self.proposals]
            self.proposals = []

            for point in inv_points:

                x = point[0]
                y = point[1]

                class_name = "car"
                print(point)
                if point[4] == 1:
                    angle_ann = "to:" + str(point[2] + point[3])
                else:
                    angle_ann = "from:" + str(point[2] + point[3])
                print(angle_ann)
                angle = self.angle_ann_2_angle(angle_ann)
                
                fg_obj = self.find_fg(class_name, angle)
                if fg_obj is not None:
                    print("fg found")
                    print("p angle: %d " % angle)


                    self.new_fg(x, y, path=fg_obj["path"].replace(".json", ".png"))

                    delta = self.delta_T # + up - down
                    # y = L
                    if y  > self.img_bg.height() // 2  - delta:
                        y_t = y - (self.img_bg.height() // 2  - delta)
                    else:
                        y_t = 1
                        
                    scale_L = (y_t) / (self.scaling_fg.scale_1_L - (self.img_bg.height() // 2 - delta))

                    scale_abs = self.scaling_fg.scale_abs / self.curr_fg.scale_abs
                    #print(scale_abs)
                    #print(scale_L)
                    scale = self.scaling_fg.scale_1_H * scale_L * scale_abs
                    #print(scale)
                    self.curr_fg.scale = scale

                    h = self.curr_fg.img.height * self.curr_fg.scale
                    self.render_fg((x, y- h//2))    
                else:
                    print("NO FG for the point p angle: %d" % angle)

            self.curr_fg = None
            
        else:
            [self.canvas.delete(i) for i in self.proposals]
            self.proposals = []

            flat_list = [item for points in inv_points for item in points]
            #print(flat_list)
            for point in inv_points:
                self.proposals.append(self.canvas.create_text(point[0],point[1]+5,fill="darkblue",font="Times 10 italic bold", text="30"))
                self.proposals.append(self.canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="red"))
        

    def set_M(self, M):
        self.M = M

    def set_delta_T(self, delta_T):
        self.delta_T = delta_T

    #########################

    def __init__(self, master, parent, pathList=None, fg_pathlist=None):
        super().__init__(master=master)

        self.nav_cb = None

        self.current_dst = None
        self.json_cmd = None

        #TODO if pathList is None: empty image
        self.idx = 0
        self.pathList = pathList
        path = str(self.pathList[self.idx])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        #ret.append(ImageTk.PhotoImage(image))

        self.winfo_toplevel().title(path)
        self.img_bg_path = path
        self.img_bg = image

        self.canvas = Canvas(self,  width=self.img_bg.width(), 
                                    height=self.img_bg.height())

        self.img_bg_tag = self.canvas.create_image(0, 0, anchor=NW, image=self.img_bg)
 
        self.fg_arr = []
            
        self.curr_fg = None
        self.sel_fg_path = None

        self.canvas.pack(fill=BOTH, expand=True)

        frm = self.canvas
        frm.bind("<Button-1>", self.on_press)
        frm.bind("<Button-2>", self.on_press)
        frm.bind("<Button-3>", self.on_press)
        frm.bind("<Button-4>", self.on_mousewheel)
        frm.bind("<Button-5>", self.on_mousewheel)
        frm.bind("<Key>", self.on_press_key)
        frm.bind("<Motion>", self.on_mousemove, add=True) # add ?

        self.canvas.focus_set()

        self.mkt_path = 'untitled.mkt'
        #self.open_mkt()

        self.M = None
        self.scaling_fg = None
        self.delta_T = 0

        self.proposals = []

        self.all_fgs = []
        self.fg_pathlist=fg_pathlist
        for file in fg_pathlist:
            json_path = str(file).replace(".png", ".json")

            try:
                with open(json_path) as json_file:  
                    try:
                        obj = json.load(json_file)
                        self.all_fgs.append( { "path" :  json_path, "json" : obj } )
                    except json.decoder.JSONDecodeError:
                        print("Invalid json for: %s" % json_path)

            except FileNotFoundError:
                # default json
                fg_json={"class_name":"car"}
                print('default %s' % fg_json)
                print("no json for: %s" % json_path)

        print(self.all_fgs)

        self.find_fg("car", 220)

    def find_fg(self, class_name, view_angle, light=None):
        filtered_fps = []
        #print(class_name)
        for fg in self.all_fgs:
            #print(fg["json"]["class_name"])
            try:
                #print("angle")
                #print(fg["json"]["view"])
                angle = self.angle_ann_2_angle(fg["json"]["view"])
                #print(angle)
            except KeyError:
                #print("KeyError")
                continue

            if fg["json"]["class_name"] == class_name and \
                self.angle_is_in_range(angle, ANGLE_DELTA, view_angle) and \
                (light == None or (light != None and fg["json"]["light"] == light) ): 
                filtered_fps.append(fg)
        
        print(filtered_fps)
        if len(filtered_fps) == 0:
            return None

        return filtered_fps[0]

    def on_mousewheel(self, event):
        s = event.state
        ctrl = (s & 0x4) != 0
        alt_l = (s & 0x8) != 0
        shift = (s & 0x1) != 0

        def delta(event):
            if event.num == 5 or event.delta < 0:
                return -1 
            return 1 

        if ctrl == True:
            act='rotate'
            count = delta(event)
        elif alt_l == True:
            count = None
            act=None
            self.shift_zorder(delta(event))
        elif shift == True:
            count = None
            act=None
        else:
            act='resize'
            count = delta(event)

        self.render_fg(delta=count, act=act)

    def open_bg(self, path):
        # reset
        for child in self.canvas.find_all():
            #print(child)
            #print(self.scaling_fg)
            if child != self.img_bg_tag and \
                (self.scaling_fg is None or (self.scaling_fg is not None and child != self.scaling_fg.img_tag)):
                    self.canvas.delete(child)

        self.fg_arr = []

        self.winfo_toplevel().title(path)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.img_bg_path = path
        self.img_bg = image

        self.canvas.itemconfig(self.img_bg_tag, image=image)
        #self.img_bg_tag = self.canvas.create_image(0, 0, anchor=NW, image=self.img_bg)

        if self.scaling_fg is not None:
            self.canvas.tag_raise(self.scaling_fg.img_tag)

    def on_press_key(self, event):
        c = event.keysym
        s = event.state

        ctrl = (s & 0x4) != 0
        #print(ctrl)
        #print(s)
        #print(c)
        if c == "KP_Delete":
            self.delete_fg()

        if c == "Left":
            if self.curr_fg is None:
                #print("left")
                if self.idx > 0:
                    self.idx -= 1
                    self.open_bg(str(self.pathList[self.idx]))
        if c == "Right":
            if self.curr_fg is None:
                #print("right")
                if self.idx < len(self.pathList) -1:
                    self.idx += 1
                    self.open_bg(str(self.pathList[self.idx]))
        
        if self.nav_cb is not None:
            self.nav_cb(event)

    def on_press(self, event):
        delta = None
        act = None
        if event.type in [EventType.ButtonPress] and event.num == 1:
            self.canvas.focus_set()

        elif event.type in [EventType.ButtonPress] and event.num == 3:
            # TODO cntxt menu
            act = 'mirror'

        elif event.type in [EventType.ButtonPress] and event.num == 2:
            self.curr_fg = None
            self.sel_fg_path = None
            
        self.render_fg(pt=(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)), delta=delta, act=act)

    def on_mousemove(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasx(event.y)


        if self.curr_fg is None and self.sel_fg_path is not None:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.new_fg(x, y)

        if self.curr_fg is not None and \
            self.curr_fg != self.scaling_fg and \
            self.scaling_fg is not None:

            # TODO autoscaling
            h = self.curr_fg.img.height * self.curr_fg.scale

            #print(self.delta_T)
            delta = self.delta_T # + up - down
            # y = L
            if y + h//2 > self.img_bg.height() // 2  - delta:
                y_t = y + h//2 - (self.img_bg.height() // 2  - delta)
            else:
                y_t = 1
                        
            scale_L = y_t / (self.scaling_fg.scale_1_L - (self.img_bg.height() // 2 - delta))

            scale_abs = self.scaling_fg.scale_abs / self.curr_fg.scale_abs
            #print(scale_abs)
            #print(scale_L)
            scale = self.scaling_fg.scale_1_H * scale_L * scale_abs 
            #print(scale)
            self.curr_fg.scale = scale
            #self.render_fg(delta=0, act="resize")


        self.render_fg((x, y))

    def mkt_traj_to_json(self, filename=None):
        cmd = {}

        #TODO all pathes as relative inside the project
        #make mkt parser to fix all pathes in ready mkt s
        if filename is None :
            mkt_path = self.mkt_path
        else :
            mkt_path = filename
        cmd['general'] = { 'mkt_path' : mkt_path }
        cmd['bg'] = {   'im_path': self.img_bg_path,
                        'inst_path': self.img_bg_path.replace('/image_2/', '/instance/'),
                        'segm_path': self.img_bg_path.replace('/image_2/', '/semantic/') 
        }

        cmd['traj'] = []

        print(cmd)
        #cmd = TextEdit(self, cmd)
        return cmd

    def mkt_to_json(self, filename=None):
        cmd = {}

        #TODO all pathes as relative inside the project
        #make mkt parser to fix all pathes in ready mkt s
        if filename is None :
            mkt_path = self.mkt_path
        else :
            mkt_path = filename
        cmd['general'] = { 'mkt_path' : mkt_path }
        cmd['bg'] = {   'im_path': self.img_bg_path,
                        'inst_path': self.img_bg_path.replace('/image_2/', '/instance/'),
                        'segm_path': self.img_bg_path.replace('/image_2/', '/semantic/') 
        }

        cmd['fgs'] = []
        for fg in self.fg_arr:
            fg_el = {}
            fg_el['x'] = int(fg.x)
            fg_el['y'] = int(fg.y)
            fg_el['class_name'] = fg.class_name
            #fg_el['img_tag'] = fg.img_tag
            fg_el['path'] = fg.path.replace('.png', '.jpg')
            fg_el['angle'] = fg.angle
            fg_el['scale'] = fg.scale
            fg_el['mirror'] = fg.mirror
            cmd['fgs'].append(fg_el)

        print(cmd)
        #cmd = TextEdit(self, cmd)
        return cmd

    def open_mkt(self):
        #FG_SRC_PATH = tkinter.filedialog.askdirectory(initialdir=FG_SRC_PATH)
        mkt_file = filedialog.askopenfilename(initialdir=MKT_ROOT, title="Select file", filetypes=(("mkt files","*.mkt"),("all files","*.*")))

        with open(mkt_file) as json_file:  
            cmd = json.load(json_file)
            print(cmd)

            self.mkt_path = mkt_file
            #def mkt_from_json(self):
            path = cmd['bg']['im_path']
            self.open_bg(path)
            fgs = cmd['fgs']
            for k in fgs:
                fg = Fg(self, 
                    k['x'], 
                    k['y'], 
                    k['path'].replace('.jpg', '.png'), 
                    k['scale'], 
                    k['angle'],
                    k['mirror'])
                self.fg_arr.append(fg)

    def save_mkt(self, filename='./untitled.mkt'):
        data = self.mkt_to_json(filename)
        with open(filename, 'w') as outfile:  
            json.dump(data, outfile)
        self.mkt_path = filename

    def save_as_mkt(self):
        filename = filedialog.asksaveasfilename(initialdir=MKT_ROOT, title="Save file", filetypes=(("mkt files","*.mkt"),("all files","*.*")))
        self.save_mkt(filename)
        self.mkt_path = filename
    
    def blend(self):
        data = self.mkt_to_json()
        print("blend")
        blender.create_image_anno(data)

    def blend_mkt_folder(self):
        print("BLEND folder START")
        #mkt_path = filedialog.askdirectory(initialdir=MKT_ROOT)
        mkt_path = filedialog.askdirectory()
        mkt_files = [os.path.join(mkt_path, f) for f in listdir(mkt_path) if isfile(join(mkt_path, f)) and f.endswith(".mkt")]

        for mkt_file in mkt_files:
            with open(mkt_file) as json_file:  
                data = json.load(json_file)
                print(data)
                print("blend %s" % json_file.name)
                blender.create_image_anno(data)

        print("BLEND folder DONE")

class UI_manager:

    def __init__(self):
        # TODO cfg / options

        self.win_main = PanelImg(master=root, parent=self, pathList=pathlist, fg_pathlist=fg_pathlist)
        self.win_main.pack(side=tk.TOP, padx=10, pady=10)

        #self.tb = ToolBar(master=root, parent=self.win_main)
        
        #TODO use fg_pathlist
        self.pf = PannelFg(master=root, parent=self.win_main, objs_path=FG_DB_ROOT)

        self.bv = PannelBV(master=root, parent=self.win_main)

# TODO config
#FG_DB_ROOT = "./FG_DB/WIERD"
FG_DB_ROOT = "./FG_DB"
DS_ROOT = "/home/tsis/Downloads/DS/K_1/data_semantics/training"
MKT_ROOT = './EXP'
im_path = DS_ROOT + '/image_2'
#inst_path = DS_ROOT + '/instance'

fg_pathlist = list(Path(FG_DB_ROOT).glob('*.png'))
pathlist = list(Path(im_path).glob('*.png'))

root = tk.Tk()
ui = UI_manager()

# TODO review to App
menu = tk.Menu(root) 
root.config(menu=menu) 
filemenu = tk.Menu(menu) 
menu.add_cascade(label='File', menu=filemenu) 
#filemenu.add_command(label='New') 
filemenu.add_command(label='Open maket...', command=ui.win_main.open_mkt) 
filemenu.add_command(label='Save maket', command=ui.win_main.save_mkt) 
filemenu.add_command(label='Save maket as...', command=ui.win_main.save_as_mkt) 
filemenu.add_separator() 
filemenu.add_command(label='Exit', underline=1, command=root.quit, accelerator="Ctrl+X") 
def bye(event):
    print("bye")
    sys.exit(0)
root.bind_all("<Control-x>", bye)

configmenu = tk.Menu(menu) 
menu.add_cascade(label='Configure', menu=configmenu) 
configmenu.add_command(label='Options...') 

toolsmenu = tk.Menu(menu) 
menu.add_cascade(label='Tools', menu=toolsmenu) 
toolsmenu.add_command(label='Blend', underline=0, command=ui.win_main.blend, accelerator="Ctrl+B") 
toolsmenu.add_command(label='Blend FOLDER', underline=0, command=ui.win_main.blend_mkt_folder, accelerator="Ctrl+Shift+B") 
def blend(event):
    ui.win_main.blend()
def blend_mkt_folder(event):
    ui.win_main.blend_mkt_folder()

root.bind_all("<Control-b>", blend)
root.bind_all("<Control-Shift-b>", blend_mkt_folder)


helpmenu = tk.Menu(menu) 
menu.add_cascade(label='Help', menu=helpmenu) 
helpmenu.add_command(label='About') 

def on_destroy():
    print("WM_DELETE_WINDOW")
    sys.exit(0)

root.protocol("WM_DELETE_WINDOW", on_destroy)

root.mainloop()

# Edit
# Undo
# Redo
# tools
#blend
#bird view
#cutters
#Segment / FG pos proposals


"""            
class ToolBar:

    def not_implemented(self):
        print("not_implemented")

    def __init__(self, master, parent):
        top = tk.Toplevel(master)

        top.winfo_toplevel().title('TOOL')
        def cancelCommand(event=None): pass
        top.protocol( 'WM_DELETE_WINDOW', cancelCommand )

        bt0 = tk.Button(top, height=1, width=15, text="Re3do", command=self.not_implemented)
        bt1 = tk.Button(top, height=1, width=15, text="Undo", command=self.not_implemented)

        bt2 = tk.Button(top, height=1, width=15, text="Segment", command=self.not_implemented)
        bt3 = tk.Button(top, height=1, width=15, text="Proposals", command=self.not_implemented)

        bt4 = tk.Button(top, height=1, width=15, text="Blend", command=parent.blend)
        bt5 = tk.Button(top, height=1, width=15, text="EditJson&Blend... ", command=self.not_implemented)

        #bt0.pack()
        bt0.grid(row=0, column=0, sticky='w')
        bt1.grid(row=0, column=1, sticky='w')

        bt2.grid(row=1, column=0, sticky='w')
        bt3.grid(row=1, column=1, sticky='w')

        bt4.grid(row=2, column=0, sticky='w')
        bt5.grid(row=2, column=1, sticky='w')
"""