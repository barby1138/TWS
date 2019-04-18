
import os
from os import listdir
from os.path import isfile, join
import shutil
import traceback

import sys

import tkinter.filedialog

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

def load_image(files):
    ret = []
    for file in files:
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        ret.append(ImageTk.PhotoImage(image))
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

        onlyfiles = [os.path.join(objs_path, f) for f in listdir(objs_path) if
                 isfile(join(objs_path, f)) and f.endswith(".png")]

        print(onlyfiles)

        hscrollbar = tk.Scrollbar(top)

        self.imgs = []

        _ratio = 0.5
        for file in onlyfiles:
            image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,
                           (round(image.shape[1] * _ratio), round(image.shape[0] * _ratio)),
                           cv2.INTER_AREA)

            # convert the images to PIL format...
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            lb = PannelFg_obj(top, self, image=image, file=file)
            lb.pack(side=TOP)
            lb.bind("<Button-1>", lb.on_select)
            self.imgs.append(image)

        self.parent.sel_fg_path = onlyfiles[0]

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

        """
        # TODO to cntx menu
        bt11 = tk.Button(top, height=1, width=10, text="BlurGa", command=self.cancel)
        bt12 = tk.Button(top, height=1, width=10, text="BlurBox", command=self.cancel)
        bt13 = tk.Button(top, height=1, width=10, text="BlurPo", command=self.cancel)
        bt14 = tk.Button(top, height=1, width=10, text="ChAdjust", command=self.cancel)
        """

        #bt0.pack()
        bt0.grid(row=0, column=0, sticky='w')
        bt1.grid(row=0, column=1, sticky='w')

        bt2.grid(row=1, column=0, sticky='w')
        bt3.grid(row=1, column=1, sticky='w')

        bt4.grid(row=2, column=0, sticky='w')
        bt5.grid(row=2, column=1, sticky='w')
        """
        bt6.grid(row=2, column=2, sticky='w')
        bt7.grid(row=3, column=0, sticky='w')
        bt8.grid(row=3, column=1, sticky='w')
        bt9.grid(row=3, column=2, sticky='w')
        bt10.grid(row=3, column=3, sticky='w')
        bt5.grid(row=4, column=0, sticky='w')
        bt6.grid(row=4, column=1, sticky='w')
        """

# TODO review incapsulate here func from canvas (Ex imageread)
class Fg:
    def __init__(self, parent, img, img_tk, img_tag, x, y, path):
        self.x = x
        self.y = y
        self.img = img
        # is needed
        self.img_tk = img_tk
        self.img_tag = img_tag
        self.path = path
        self.parent = parent
        self.angle = 0.0
        self.scale = 1.0

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
        def render_curr_fg():
                w = int(self.curr_fg.img.width * self.curr_fg.scale)
                h = int(self.curr_fg.img.height * self.curr_fg.scale)
                img = self.curr_fg.img.resize((w, h), Image.ANTIALIAS)
                img = img.rotate(self.curr_fg.angle)

                self.curr_fg.img_tk = ImageTk.PhotoImage(img)
                self.canvas.itemconfig(self.curr_fg.img_tag, image = self.curr_fg.img_tk)
 
        if self.curr_fg and pt is not None:
            deltax = pt[0] - self.curr_fg.x
            deltay = pt[1] - self.curr_fg.y
            self.canvas.move(self.curr_fg.img_tag, deltax, deltay)
            self.curr_fg.x += deltax
            self.curr_fg.y += deltay

        if self.curr_fg and delta and act is not None:
            if act == 'resize':
                scale = delta * 0.01
                self.curr_fg.scale += scale
                render_curr_fg()
            elif act == 'rotate':
                angle = delta
                self.curr_fg.angle += angle
                render_curr_fg()

        if self.curr_fg is not None and act == 'mirror':
            self.curr_fg.img = ImageOps.mirror(self.curr_fg.img)
            render_curr_fg()
            
        if self.curr_fg and path is not None and act == 'update':
            # TODO check if path the same
            self.curr_fg.img = Image.open(path)
            self.curr_fg.path = path
            render_curr_fg()

    def delete_fg(self):
        if self.curr_fg is not None:
                a = self.fg_arr.index(self.curr_fg)
                self.canvas.delete(self.fg_arr[a].img_tag)
                del self.fg_arr[a]
                self.curr_fg = None

    def update_fg(self, new_path):
        #print(new_path)
        # return focus to handle key press
        self.canvas.focus_set()
        self.sel_fg_path = new_path
        act = 'update'
        self.render_fg(act=act, path=new_path)

    def new_fg(self, x, y):
        # TODO move to fg class
        path = self.sel_fg_path

        def make_square(im, fill_color=(255, 255, 255, 0)):
            x,y = im.size
            size = max(im.size)
            print(im.size)
            print(size)
            new_im = Image.new('RGBA', (size, size), fill_color)
            new_im.paste(im, (int((size-x) / 2), int((size-y) / 2)))
            #new_im.show()
            return new_im

        img_fg = Image.open(path)
        #img_fg = make_square(img_fg)
        img_fg_tk = ImageTk.PhotoImage(img_fg)
        img_fg_tag = self.canvas.create_image(x, y, image=img_fg_tk)
        self.curr_fg = Fg(self, img_fg, img_fg_tk, img_fg_tag, x, y, path)
        self.fg_arr.append(self.curr_fg)
        self.canvas.tag_bind(img_fg_tag, '<Button-1>', self.curr_fg.on_fg_click)

    def __init__(self, master, parent, path):
        super().__init__(master=master)

        self.current_dst = None
        self.json_cmd = None

        #ret = []
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        #ret.append(ImageTk.PhotoImage(image))

        self.img_bg_path = path
        self.img_bg = image

        self.canvas = Canvas(self,  width=self.img_bg.width(), 
                                    height=self.img_bg.height())

        self.img_bg_tag = self.canvas.create_image(0, 0, anchor=NW, image=self.img_bg)

        self.fg_arr = []
        self.curr_fg = None
        # set by fg pannel selection
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

    def on_mousewheel(self, event):
        s = event.state
        ctrl = (s & 0x4) != 0
        alt_l = (s & 0x8) != 0
        shift = (s & 0x1) != 0
        
        #print(ctrl)
        #print(alt_l)
        #print(shift)

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

    def on_press_key(self, event):
        c = event.keysym
        s = event.state

        ctrl = (s & 0x4) != 0
        #print(ctrl)
        #print(s)
        #print(c)
        if c == "KP_Delete":
            #print("KP_Delete")
            self.delete_fg()
            
    def on_press(self, event):
        delta = None
        act = None
        if event.type in [EventType.ButtonPress] and event.num == 1:
            self.canvas.focus_set()

        elif event.type in [EventType.ButtonPress] and event.num == 3:
            # TODO cntxt menu
            act = 'mirror'

        elif event.type in [EventType.ButtonPress] and event.num == 2:
            if self.curr_fg is None:
                x = self.canvas.canvasx(event.x)
                y = self.canvas.canvasy(event.y)
                self.new_fg(x, y)
            else:
                self.curr_fg = None

        self.render_fg(pt=(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)), delta=delta, act=act)

    def on_mousemove(self, event):
        self.render_fg((self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)))

    def blend(self):
        cmd = {}
        cmd['bg'] = { 'path': self.img_bg_path }

        cmd['fgs'] = []
        for k in range(0,len(self.fg_arr)):
            fg_el = {}
            fg = self.fg_arr[k]
            fg_el['x'] = int(fg.x)
            fg_el['y'] = int(fg.y)
            #fg_el['img'] = fg.img
            #fg_el['img_tk'] = fg.img_tk
            #fg_el['img_tag'] = fg.img_tag
            # TODO obj_id obj_type
            fg_el['path'] = fg.path.replace('.png', '.jpg')
            fg_el['angle'] = fg.angle
            fg_el['scale'] = fg.scale
            cmd['fgs'].append(fg_el)

        print(cmd)
        #blender.gen_syn_data(cmd)
        blender.create_image_anno(cmd)

class UI_manager:

    def __init__(self):
        # TODO cfg / options

        #_cmd = None
        # TODO review set inside PanelImg
        path = str(pathlist[0])
        #imgs = load_image([path, path])

        self.win_main = PanelImg(master=root, parent=self, path=path)
        self.win_main.pack(side=tk.TOP, padx=10, pady=10)

        self.tb = ToolBar(master=root, parent=self.win_main)
        self.pf = PannelFg(master=root, parent=self.win_main, objs_path=FG_DB_ROOT)

# TODO config
FG_DB_ROOT = "./FG_DB"
DS_ROOT = "./DS"
path = DS_ROOT + '/image'
path_inst = DS_ROOT + '/instance'

pathlist = list(Path(path).glob('*.png'))

root = tk.Tk()

menu = tk.Menu(root) 
root.config(menu=menu) 
filemenu = tk.Menu(menu) 
menu.add_cascade(label='File', menu=filemenu) 
filemenu.add_command(label='New') 
filemenu.add_command(label='Open...') 
filemenu.add_command(label='Save') 
filemenu.add_command(label='Save.as..') 
filemenu.add_separator() 
filemenu.add_command(label='Exit', command=root.quit) 

configmenu = tk.Menu(menu) 
menu.add_cascade(label='Configure', menu=configmenu) 
configmenu.add_command(label='Options...') 

helpmenu = tk.Menu(menu) 
menu.add_cascade(label='Help', menu=helpmenu) 
helpmenu.add_command(label='About') 

def on_destroy():
    print("WM_DELETE_WINDOW")
    sys.exit(0)

root.protocol("WM_DELETE_WINDOW", on_destroy)

ui = UI_manager()
root.mainloop()






"""
def format_cmd(_name, _dst, _top, _btm, _sdw, _o_blr=(1, 0), _s_blr=(1, 0), _s_factor=0.8, _o_flip=-2):
    _cmd = {
        "targetPath": _dst,
        "name": _name,
        "org_inst": ORG_INST,
        "res_inst": RES_INST,
        "background": ORG_BG,
        "object": ORG_FG,
        "result": RES_BG,
        "diff": RES_DIFF,
        "augmentationPoints": {
            "top": {
                "x": _top[0],
                "y": _top[1],
            },
            "bottom": {
                "x": _btm[0],
                "y": _btm[1],
            },
            "shadow": {
                "x": _sdw[0],
                "y": _sdw[1],
            }
        },
        "objectBlur": {
            "width": _o_blr[0],
            "height": _o_blr[1],
        },
        "shadowBlur":
            {
                "width": _s_blr[0],
                "height": _s_blr[1],
            },
        "shadowIntensity": _s_factor,
        "flip": _o_flip,
        "past_points": [],
        "all_points": [],
    }
    return _cmd
"""