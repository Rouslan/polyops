
import os
import sys
import tkinter
from tkinter import ttk
import json
from collections import namedtuple, defaultdict

READ_SIZE = 0x1000

LoopPoint = namedtuple('LoopPoint','p next')
ScaledLoopPoint = namedtuple('LoopPoint','p orig_p next')

def run_plotter(file):
    data = None
    data_pend = None
    win = tkinter.Tk()
    can = tkinter.Canvas(win,bg='#FFFFFF')
    can.pack(fill=tkinter.BOTH,expand=True)
    sbar = ttk.Frame(win)
    sbar.pack(side=tkinter.BOTTOM)
    status = ttk.Label(sbar,text="thingy")
    status.pack(side=tkinter.LEFT,anchor='w')

    # a second hidden canvas for nearest point calculation
    pcan = tkinter.Canvas(win)

    p_indices = defaultdict(list)
    ci_to_pi = {}

    sel_coord = -1

    def new_data(e):
        nonlocal data

        data = json.loads(data_pend)
        for i,p in enumerate(data):
            data[i] = LoopPoint((p['p'][0],p['p'][1]),p['next'])

        can.delete('all')
        pcan.delete('all')
        p_indices.clear()

        if not data: return

        minx = maxx = data[0].p[0]
        miny = maxy = data[0].p[1]
        for p in data:
            minx = min(p.p[0],minx)
            maxx = max(p.p[0],maxx)
            miny = min(p.p[1],miny)
            maxy = max(p.p[1],maxy)

        scale = 800 / max(maxx - minx,maxy - miny)

        for i,p in enumerate(data):
            data[i] = ScaledLoopPoint(
                ((p.p[0] - minx) * scale + 40,(p.p[1] - miny) * scale + 40),
                p.p,
                p.next)
            p_indices[p.p].append(i)

        for i,p in enumerate(data):
            nextp = data[p.next]
            can.create_line(p.p,nextp.p,arrow=tkinter.LAST,tags='_'+str(i))

        for indices in p_indices.values():
            p = data[indices[0]].p
            can.create_text(p,text=str('/'.join(map(str,indices))),fill='#00aa00')
            ci_to_pi[pcan.create_rectangle(p,p,width=0)] = indices[0]

    win.bind('<<NewData>>',new_data)

    def onmove(e):
        nonlocal sel_coord

        cp = pcan.find_closest(e.x,e.y)
        new_coord = -1
        if len(cp):
            pi = ci_to_pi[cp[0]]
            p = data[pi]
            dx = p.p[0]-e.x
            dy = p.p[1]-e.y
            if (dx*dx + dy*dy) <= 25:
                new_coord = pi
        
        if new_coord != sel_coord:
            if sel_coord >= 0:
                for i in p_indices[data[sel_coord].orig_p]:
                    can.itemconfigure('_'+str(i),fill='black')
            
            if new_coord >= 0:
                sparts = []
                for i in p_indices[data[new_coord].orig_p]:
                    can.itemconfigure('_'+str(i),fill='red')
                    sparts.append('{} -> {}'.format(i,data[i].next))

                status['text'] = '{},{}: {}'.format(data[new_coord].orig_p[0],data[new_coord].orig_p[1],', '.join(sparts))
            else:
                status['text'] = ''
            
            sel_coord = new_coord


    can.bind('<Motion>',onmove)

    def file_handler(file,mask):
        nonlocal data_pend

        line = ''

        while True:
            chunk = os.read(file,READ_SIZE)
            line += str(chunk,encoding='utf-8')
            if len(chunk) < READ_SIZE: break
        
        parts = line.split('\n')
        line = parts.pop()
        if parts:
            data_pend = parts[-1]
            win.event_generate('<<NewData>>')
    
    win.tk.createfilehandler(file.fileno(),tkinter.READABLE,file_handler)

    win.mainloop()


if __name__ == '__main__':
    run_plotter(sys.stdin)