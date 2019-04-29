
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

        print(onlyfiles)

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

        self.parent.sel_fg_path = onlyfiles[0]

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

class Fg:

    def __init__(self, parent, x, y, path, scale=1.0, angle=0.0, mirror=False, mode='Load'):
        SIZE_MAX = 300

        # TODO handle FileNotFoundError?
        img = Image.open(path)
        w, h = img.width, img.height
        
        if mode == 'New' and (w > SIZE_MAX or h > SIZE_MAX):
            scale = SIZE_MAX / max(w,h)
            w_o, h_o = int(img.width*scale), int(img.height*scale)
            img_resize = img.resize((w_o, h_o), Image.ANTIALIAS)
        else:
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
            print("WARNING!!! corrupted json")
            self.class_name = 'unk'

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
                w = int(self.img.width * self.scale)
                h = int(self.img.height * self.scale)
                img = self.img.resize((w, h), Image.ANTIALIAS)
                if self.mirror == True:
                    img = ImageOps.mirror(img)
                img = img.rotate(self.angle, expand=True)

                self.img_tk = ImageTk.PhotoImage(img)
                self.parent.canvas.itemconfig(self.img_tag, image = self.img_tk)

    def render(self, pt=None, delta=None, path=None, act=None):
        if pt is not None:
            deltax = pt[0] - self.x
            deltay = pt[1] - self.y
            self.parent.canvas.move(self.img_tag, deltax, deltay)
            self.x += deltax
            self.y += deltay

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
            #self.img = ImageOps.mirror(self.img)
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
                a = self.fg_arr.index(self.curr_fg)
                self.canvas.delete(self.fg_arr[a].img_tag)
                del self.fg_arr[a]
                self.curr_fg = None

    def update_fg(self, new_path):
        self.canvas.focus_set()
        self.sel_fg_path = new_path
        self.render_fg(act='update', path=new_path)

    def new_fg(self, x, y):
        self.curr_fg = Fg(self, x, y, self.sel_fg_path, mode='New')
        self.fg_arr.append(self.curr_fg)

    def __init__(self, master, parent, pathList=None):
        super().__init__(master=master)

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

        #self.open_mkt()

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
        self.winfo_toplevel().title(path)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.img_bg_path = path
        self.img_bg = image

        self.canvas.itemconfig(self.img_bg_tag, image=image)

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

    def mkt_to_json(self):
        cmd = {}
        cmd['bg'] = { 'im_path': self.img_bg_path,
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
        data = self.mkt_to_json()
        with open(filename, 'w') as outfile:  
            json.dump(data, outfile)

    def save_as_mkt(self):
        filename = filedialog.asksaveasfilename(initialdir=MKT_ROOT, title="Save file", filetypes=(("mkt files","*.mkt"),("all files","*.*")))
        self.save_mkt(filename)
    
    def blend(self):
        #self.save_mkt()
        data = self.mkt_to_json()
        print("blend")
        blender.create_image_anno(data)

class UI_manager:

    def __init__(self):
        # TODO cfg / options

        self.win_main = PanelImg(master=root, parent=self, pathList=pathlist)
        self.win_main.pack(side=tk.TOP, padx=10, pady=10)

        #self.tb = ToolBar(master=root, parent=self.win_main)
        self.pf = PannelFg(master=root, parent=self.win_main, objs_path=FG_DB_ROOT)

# TODO config
FG_DB_ROOT = "./FG_DB"
DS_ROOT = "/home/tsis/Downloads/DS/K/data_semantics/training"
MKT_ROOT = './EXP'
im_path = DS_ROOT + '/image_2'
#inst_path = DS_ROOT + '/instance'

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
filemenu.add_command(label='Save as maket...', command=ui.win_main.save_as_mkt) 
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
def blend(event):
    ui.win_main.blend()
root.bind_all("<Control-b>", blend)

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