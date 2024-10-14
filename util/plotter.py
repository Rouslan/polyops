
import os
import operator
import enum
import tkinter as tk
from tkinter import ttk, filedialog
import json
from collections import namedtuple,defaultdict

READ_SIZE = 0x1000
ZOOM_FACTOR = 1.1

AUX_VIEW_NOTHING = 0
AUX_VIEW_STATE = 1
AUX_VIEW_INDEX = 2

class MyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,Vec):
            return list(obj)
        if isinstance(obj,ScaledLoopPoint):
            return {'p':obj.p,'next':obj.next,'state':obj.state.value,'loop_i':obj.loop_i}

        super().default(obj)

class Vec(namedtuple('_VecBase','x y')):
    __slots__ = ()

    def __new__(cls,*args):
        if len(args) == 2:
            return super().__new__(cls,*args)
        if len(args) == 1:
            if isinstance(args[0],Vec):
                return args[0]
            if hasattr(type(args[0]),'__iter__'):
                return super().__new__(cls,*args[0])
            return super().__new__(cls,args[0],args[0])

        raise TypeError('Vec takes 1 or 2 arguments')

    def __add__(self,b):
        return Vec(map(operator.add,self,Vec(b)))

    __radd__ = __add__

    def __sub__(self,b):
        return Vec(map(operator.sub,self,Vec(b)))

    def __rsub__(self,a):
        return Vec(map(operator.add,Vec(a),self))

    def __mul__(self,b):
        return Vec(map(operator.mul,self,Vec(b)))

    __rmul__ = __mul__

    def __truediv__(self,b):
        return Vec(map(operator.truediv,self,Vec(b)))

    def __rtruediv__(self,a):
        return Vec(map(operator.truediv,Vec(a),self))

    def __floordiv__(self,b):
        return Vec(map(operator.floordiv,self,Vec(b)))

    def __rfloordiv__(self,a):
        return Vec(map(operator.floordiv,Vec(a),self))

    @staticmethod
    def map(fun,*args):
        return Vec(map(fun,*args))

PADDING = Vec(40,40)

class LineState(enum.Enum):
    undef = 'undef'
    check = 'check'
    discard = 'discard'
    keep = 'keep'
    keep_rev = 'keep_rev'
    invalid = '<invalid>'

class ScaledLoopPoint:
    def __init__(self,p,next,state,loop_i,scaled_p = None):
        self.p = Vec(p)
        self.next = next
        self.state = state
        self.loop_i = loop_i
        self.scaled_p = scaled_p and Vec(scaled_p)

def run_plotter(file):
    data = None
    data_pend = None
    win = tk.Tk()
    win.style = ttk.Style()
    win.style.theme_use('clam')
    win.rowconfigure(0,weight=1)
    win.columnconfigure(0,weight=1)
    can = tk.Canvas(win,bg='#FFFFFF',confine=False)
    can.grid(row=0,column=0,sticky=tk.N+tk.E+tk.S+tk.W)
    sbar = ttk.Frame(win)
    sbar.grid(row=2,column=0,columnspan=2)
    status = ttk.Label(sbar,text="thingy")
    status.pack(side=tk.LEFT,anchor='w')

    scroll_x = ttk.Scrollbar(win,orient='horizontal',command=can.xview)
    scroll_x.grid(row=1,column=0,sticky=tk.W+tk.E)
    scroll_y = ttk.Scrollbar(win,orient='vertical',command=can.yview)
    scroll_y.grid(row=0,column=1,sticky=tk.N+tk.S)

    can.configure(xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)

    # a second hidden canvas for nearest point calculation
    pcan = tk.Canvas(win)

    p_indices = defaultdict(list)
    ci_to_pi = {}

    sel_coord = -1

    scale = 1
    minc = None
    maxc = None
    unpadded_area = Vec(0)

    aux_view_ctrl = tk.IntVar(value=AUX_VIEW_NOTHING)
    view_hline = tk.IntVar()
    view_vline = tk.IntVar()

    hline_id = None
    vline_id = None

    # the canvas coordinates of the cursor or None
    mouse_p = None

    def build_scene():
        nonlocal unpadded_area, hline_id, vline_id

        aux_view = aux_view_ctrl.get()

        can.delete('all')
        pcan.delete('all')
        p_indices.clear()
        hline_id = None
        vline_id = None

        if not data: return

        assert maxc is not None
        assert minc is not None

        sfactor = 800 / max(maxc - minc) * scale

        for i,p in enumerate(data):
            p.scaled_p = (p.p - minc) * sfactor + PADDING
            p_indices[p.p].append(i)

        for i,p in enumerate(data):
            nextp = data[p.next]
            can.create_line(p.scaled_p,nextp.scaled_p,arrow=tk.LAST,tags='_'+str(i))
        
        if aux_view != AUX_VIEW_NOTHING:
            for p in data:
                if aux_view == AUX_VIEW_INDEX:
                    text = str(p.loop_i)
                else:
                    text = p.state.value
                can.create_text((p.scaled_p + data[p.next].scaled_p)/2, text=text, fill='#3333aa')

        for indices in p_indices.values():
            p = data[indices[0]].scaled_p
            can.create_text(p,text=str('/'.join(map(str,indices))),fill='#00aa00')
            ci_to_pi[pcan.create_rectangle(p,p,width=0)] = indices[0]

        unpadded_area = (maxc - minc) * sfactor
        bottom_right = unpadded_area + PADDING*2
        can.configure(scrollregion=(
            0,
            0,
            bottom_right.x,
            bottom_right.y))

        return bottom_right
    
    def update_guide_lines():
        nonlocal hline_id, vline_id

        cx1 = can.canvasx(0)
        cx2 = can.canvasx(can.winfo_width())
        cy1 = can.canvasy(0)
        cy2 = can.canvasy(can.winfo_height())

        if view_hline.get() and mouse_p is not None:
            coords = (cx1,mouse_p.y,cx2,mouse_p.y)
            if hline_id is None:
                hline_id = can.create_line(*coords)
            else:
                can.coords(hline_id,*coords)
        elif hline_id is not None:
            can.delete(hline_id)
            hline_id = None

        if view_vline.get() and mouse_p is not None:
            coords = (mouse_p.x,cy1,mouse_p.x,cy2)
            if vline_id is None:
                vline_id = can.create_line(*coords)
            else:
                can.coords(vline_id,*coords)
        elif vline_id is not None:
            can.delete(vline_id)
            vline_id = None
    
    def update_data():
        build_scene()
        update_guide_lines()

    def new_data(e):
        nonlocal data, minc, maxc

        assert data_pend is not None

        data = json.loads(data_pend)
        for i,p in enumerate(data):
            data[i] = ScaledLoopPoint(
                p['p'],
                p['next'],
                LineState(p['state']),
                p['loop_i'])

        if data:
            minc = maxc = data[0].p
            for p in data:
                minc = Vec.map(min,p.p,minc)
                maxc = Vec.map(max,p.p,maxc)

        update_data()

    win.bind('<<NewData>>',new_data)

    def onmove(e):
        nonlocal sel_coord, mouse_p

        if not data: return

        mouse_p = Vec(can.canvasx(e.x),can.canvasy(e.y))
        cp = pcan.find_closest(*mouse_p)
        new_coord = -1
        if len(cp):
            pi = ci_to_pi[cp[0]]
            p = data[pi]
            delta = p.scaled_p - mouse_p
            if (delta.x*delta.x + delta.y*delta.y) <= 25:
                new_coord = pi

        if new_coord != sel_coord:
            if sel_coord >= 0:
                for i in p_indices[data[sel_coord].p]:
                    can.itemconfigure('_'+str(i),fill='black')

            if new_coord >= 0:
                sparts = []
                for i in p_indices[data[new_coord].p]:
                    can.itemconfigure('_'+str(i),fill='red')
                    sparts.append('{} -> {}'.format(i,data[i].next))

                status['text'] = '{},{}: {}'.format(data[new_coord].p[0],data[new_coord].p[1],', '.join(sparts))
            else:
                status['text'] = ''

            sel_coord = new_coord
        
        update_guide_lines()


    can.bind('<Motion>',onmove)

    def onscroll(e):
        nonlocal scale

        if unpadded_area.x == 0 or unpadded_area.y == 0: return

        c = (Vec(can.canvasx(e.x),can.canvasy(e.y)) - PADDING) / unpadded_area

        if e.num == 4: scale *= ZOOM_FACTOR
        else: scale /= ZOOM_FACTOR

        full_area = build_scene()

        new_c0 = c * unpadded_area + PADDING - (e.x,e.y)
        scroll = new_c0 / full_area

        can.xview_moveto(scroll.x)
        can.yview_moveto(scroll.y)

        update_guide_lines()

    can.bind('<Button-4>',onscroll)
    can.bind('<Button-5>',onscroll)

    def onsave(e):
        if not data: return

        f = filedialog.asksaveasfile(parent=win,defaultextension='.txt')
        if f is None: return

        with f:
            json.dump(data,f,cls=MyEncoder,separators=(',',':'))
            f.write('\n')
    win.bind('<Control-KeyPress-s>',onsave)

    line = ''
    def file_handler(file,mask):
        nonlocal data_pend, line

        chunk = os.read(file,READ_SIZE)
        if len(chunk):
            line += str(chunk,encoding='utf-8')
            while len(chunk) == READ_SIZE:
                chunk = os.read(file,READ_SIZE)
                line += str(chunk,encoding='utf-8')

            parts = line.rsplit('\n',maxsplit=2)
            line = parts.pop()

            if parts:
                data_pend = parts[-1]
                win.event_generate('<<NewData>>')
        else:
            win.tk.deletefilehandler(file)
            if len(line):
                data_pend = line
                win.event_generate('<<NewData>>')

    win.tk.createfilehandler(file,tk.READABLE,file_handler)

    menu = tk.Menu(win)

    view_menu = tk.Menu(menu,tearoff=False)
    view_menu.add_radiobutton(label='Hide Aux',value=AUX_VIEW_NOTHING,variable=aux_view_ctrl,command=update_data)
    view_menu.add_radiobutton(label='Show Aux State',value=AUX_VIEW_STATE,variable=aux_view_ctrl,command=update_data)
    view_menu.add_radiobutton(label='Show Aux Loop Index',value=AUX_VIEW_INDEX,variable=aux_view_ctrl,command=update_data)
    view_menu.add_separator()
    view_menu.add_checkbutton(label='Show Horizontal Guide',variable=view_hline,command=update_guide_lines)
    view_menu.add_checkbutton(label='Show Vertical Guide',variable=view_vline,command=update_guide_lines)
    menu.add_cascade(label='View',menu=view_menu)

    win.config(menu=menu)
    win.mainloop()


if __name__ == '__main__':
    run_plotter(0)
