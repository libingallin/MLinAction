# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:18:52 2017

@author: libing
"""

import Tkinter

import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import regTree


matplotlib.use('TkAgg')


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = regTree.createTree(reDraw.rawDat, regTree.modelLeaf,
                                    regTree.modelErr, (tolS, tolN))
        yHat = regTree.createForecast(myTree, reDraw.testDat,
                                      regTree.modelTreeEval)
    else:
        myTree = regTree.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTree.createForecast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.raw[:, 0], reDraw.rawDat[:, 1], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolNentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)


# This creates a toplevel widget of Tk which usually is the main window
# of an application
root = Tkinter.Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Tkinter.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)

Tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')

Tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Tkinter.Button(root, text='ReDraw', command=drawNewTree). \
                                grid(row=1, column=2, rowspan=3)

chkBtnVar = Tkinter.IntVar()
chkBtnVar = Tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtnVar.grid(row=3, column=0, columnspan=2)


reDraw.rawDat = np.mat(regTree.loadDataSet('sine.txt'))
reDraw.rawData = np.arange(min(reDraw.rawDat[:, 0]),
                           max(reDraw.rawDat[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()
