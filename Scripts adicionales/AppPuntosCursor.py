import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from tkinter import messagebox as mb
from os import scandir, listdir
import os


class Aplicacion:
    def __init__(self):
        self.a = ""
        self.ventana = tkinter.Tk()
        self.ventana.geometry('500x300')
        self.agregar_menu()
        self.ventana.wm_title("Taller")
        self.posicion = [0]
        self.validar = False
        self.press = False  # Flag to track mouse press
        self.selected_point = None  # Initialize selected point to None
        self.ventana.mainloop()

    def agregar_menu(self):
        menubar1 = tkinter.Menu(self.ventana)
        self.ventana.config(menu=menubar1)
        opciones1 = tkinter.Menu(menubar1, tearoff=0)
        opciones1.add_command(label="Seleccionar directorio",
                              command=self.setDirectory)
        opciones1.add_command(label="Guardar archivo", command=self.guardar)
        opciones1.add_separator()
        opciones1.add_command(label="Salir", command=self.salir)
        menubar1.add_cascade(label="Archivo", menu=opciones1)

        btnNext = tkinter.Button(
            master=self.ventana, text="Siguiente", command=self.onNext)
        btnNext.pack(side=tkinter.BOTTOM)
        btnPrev = tkinter.Button(
            master=self.ventana, text="Atrás", command=self.onPrevious)
        btnPrev.pack(side=tkinter.BOTTOM)

        self.lblDir = tkinter.Label(master=self.ventana)
        self.lblDir.pack()
        self.btnSelect = tkinter.Button(
            master=self.ventana, text="Seleccionar Imagen", command=self.openImg)
        self.btnSelect.pack(side=tkinter.TOP)
        self.btnSelect.pack_forget()
        label = tkinter.Label(master=self.ventana, text="Use las ← ↑ ↓ → del teclado para mover los puntos")
        label.place(relx=0.7, rely=0.95)

    def update(self):
        self.a.clear()
        self.modificacion = self.a.scatter(
            self.ejeX, self.ejeY, s=5, picker=1, c=self.color)
        self.a.imshow(self.img)
        self.posicion = [0]
        self.a.grid(True)
        self.canvas.draw()

    def salir(self):
        sys.exit()

    def guardar(self):
        np.savetxt(self.directory + "/" + self.list[self.posImage + 1], np.transpose([self.ejeX, self.ejeY]),
                   delimiter=" ", header="version: 1\nn_points: 68\n{", footer="}", fmt="%i", comments='')
        mb.showinfo("Información", "Los datos fueron guardados correctamente.")
        self.validar = False
        self.update()

    def onpick(self, event):
        ind = event.ind
        if self.modificacion.get_facecolor()[self.posicion[0]][0] == 0:
            self.modificacion.get_facecolor()[self.posicion[0]] = [
                0, 0.50196078, 0, 1, ]
            self.modificacion.get_edgecolors()[self.posicion[0]] = [
                0, 0.50196078, 0, 1, ]

        self.modificacion.get_facecolor()[ind[0]] = [
            0, 1, 1, 1, ]
        self.modificacion.get_edgecolors()[ind[0]] = [
            0, 1, 1, 1, ]
        self.canvas.draw()
        self.posicion = ind

    def cambios(self):
        x, y = self.ejeX[self.posicion[0]], self.ejeY[self.posicion[0]]
        x = max(0, min(x, self.img.shape[1] - 1))  # Ensure x is within image width
        y = max(0, min(y, self.img.shape[0] - 1))  # Ensure y is within image height
        self.modificacion.get_offsets().data[self.posicion[0]] = [x, y]

        self.modificacion.get_facecolor()[self.posicion[0]] = [
            1, 0, 0, 1, ]
        self.modificacion.get_edgecolors()[self.posicion[0]] = [
            1, 0, 0, 1, ]
        self.canvas.draw()
        self.validar = True

    def left(self, event):
        self.ejeX[self.posicion[0]] = int(self.ejeX[self.posicion[0]]) - 1
        self.cambios()

    def right(self, event):
        self.ejeX[self.posicion[0]] = self.ejeX[self.posicion[0]] + 1
        self.cambios()

    def up(self, event):
        self.ejeY[self.posicion[0]] = self.ejeY[self.posicion[0]] - 1
        self.cambios()

    def down(self, event):
        self.ejeY[self.posicion[0]] = self.ejeY[self.posicion[0]] + 1
        self.cambios()

    def setDirectory(self):
        directory = fd.askdirectory(
            initialdir="/", title="Seleccione una carpeta",)

        ls = [arch.name for arch in scandir(directory) if arch.is_file()]
        self.list = ls
        self.allImgs = [img for img in ls if img.endswith(".jpg")]
        self.posImage = 0
        self.directory = directory
        self.n = 1
        label = "Imagen " + str(self.n) + "/" + str(len(self.allImgs)
                                                    ) + ": " + self.list[self.posImage]
        self.lblDir.config(text=label)
        self.btnSelect.pack(side=tkinter.TOP)
        if self.a != "":
            self.a.clear()
            self.update()
        else:
            self.recuperar(self.directory + '/' + self.list[self.posImage])
        self.respuesta = False

    def onNext(self):
        if self.posImage == (len(self.list) - 2):
            mb.showinfo("Aviso",
                        "No hay más imágenes")
            return
        if self.validar == True:
            self.respuesta = mb.askyesno(
                message="¿Desea guardar los cambios?", title="IA")

        if self.respuesta == True:
            np.savetxt(self.directory + "/" + self.list[self.posImage + 1], np.transpose([self.ejeX, self.ejeY]),
                       delimiter=" ", header="version: 1\nn_points: 68\n{", footer="}", fmt="%i", comments='')

        self.respuesta = False
        self.validar = False
        self.posImage = self.posImage + 2
        self.n = self.n + 1
        label = "Imagen " + str(self.n) + "/" + str(len(self.allImgs)
                                                    ) + ": " + self.list[self.posImage]
        self.lblDir.config(text=label)
        self.img = mpimg.imread(self.directory + '/' + self.list[self.posImage])
        self.puntos = np.loadtxt(self.directory + '/' + self.list[self.posImage + 1], comments=(
            'version:', 'n_points:', '{', '}'))
        self.ejeX = []
        self.ejeY = []
        for i in self.puntos:
            self.ejeX.append(i[0])
            self.ejeY.append(i[1])
        if hasattr(self, 'fig'):
            self.update()

    def onPrevious(self):
        if self.posImage == 0:
            mb.showinfo("Aviso",
                        "No hay más imágenes")
            return
        if self.validar == True:
            self.respuesta = mb.askyesno(
                message="¿Desea guardar los cambios?", title="IA")

        if self.respuesta == True:
            np.savetxt(self.directory + "/" + self.list[self.posImage + 1], np.transpose([self.ejeX, self.ejeY]),
                       delimiter=" ", header="version: 1\nn_points: 68\n{", footer="}", fmt="%i", comments='')

        self.respuesta = False
        self.validar = False
        self.posImage = self.posImage - 2
        self.n = self.n - 1
        label = "Imagen " + str(self.n) + "/" + str(len(self.allImgs)
                                                    ) + ": " + self.list[self.posImage]
        self.lblDir.config(text=label)
        self.img = mpimg.imread(self.directory + '/' + self.list[self.posImage])
        self.puntos = np.loadtxt(self.directory + '/' + self.list[self.posImage + 1], comments=(
            'version:', 'n_points:', '{', '}'))
        self.ejeX = []
        self.ejeY = []
        for i in self.puntos:
            self.ejeX.append(i[0])
            self.ejeY.append(i[1])
        if hasattr(self, 'fig'):
            self.update()

    def recuperar(self, ruta):
        if ruta != '':
            self.puntos = np.loadtxt(self.directory + '/' + self.list[self.posImage + 1], comments=(
                'version:', 'n_points:', '{', '}'))
            self.img = mpimg.imread(ruta)
            self.ejeX = []
            self.ejeY = []
            for i in self.puntos:
                self.ejeX.append(i[0])
                self.ejeY.append(i[1])
            self.fig = Figure(figsize=(5, 5))
            self.a = self.fig.add_subplot(111)
            self.color = []
            for i in range(68):
                self.color.append("green")
            self.modificacion = self.a.scatter(
                self.ejeX, self.ejeY, s=5, picker=1, color=self.color)
            self.a.set_ylabel("Y", fontsize=14)
            self.a.set_xlabel("X", fontsize=14)
            self.a.grid(True)
            self.a.imshow(self.img)
            # CREAR AREA DE DIBUJO DE TKINTER.
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.ventana)
            self.fig.canvas.mpl_connect('pick_event', self.onpick)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(
                self.canvas, self.ventana)  # barra de iconos
            toolbar.update()
            self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
            self.ventana.bind("<Left>", self.left)
            self.ventana.bind("<Right>", self.right)
            self.ventana.bind("<Up>", self.up)
            self.ventana.bind("<Down>", self.down)

            # Connect mouse events
            self.canvas.mpl_connect('button_press_event', self.mouse_press)
            self.canvas.mpl_connect('button_release_event', self.mouse_release)
            self.canvas.mpl_connect('motion_notify_event', self.mouse_move)

            return 1

    def openImg(self):
        rutaImg = fd.askopenfilename(initialdir=self.directory, title="Seleccione archivo", filetypes=(
            ("txt files", "*.JPG"), ("todos los archivos", "*.*")))
        nameImg = os.path.basename(rutaImg)
        dirImg = os.path.dirname(rutaImg)
        self.path = dirImg
        print("DIR:", dirImg, "ruta", rutaImg)
        lstFiles = [arch.name for arch in scandir(
            self.path) if arch.is_file()]

        self.imgs = [img for img in lstFiles if img.endswith(".jpg")]
        self.filePts = [img for img in lstFiles if img.endswith(".pts")]

        for i in self.imgs:
            self.pos = self.imgs.index(nameImg)
            break

        self.img = mpimg.imread(self.directory + '/' + self.imgs[self.pos])
        self.puntos = np.loadtxt(self.directory + '/' + self.filePts[self.pos], comments=(
            'version:', 'n_points:', '{', '}'))
        self.ejeX = []
        self.ejeY = []
        for i in self.puntos:
            self.ejeX.append(i[0])
            self.ejeY.append(i[1])
        if hasattr(self, 'fig'):
            self.update()
        self.n = self.pos
        self.n += 1
        label = "Imagen " + str(self.n) + "/" + \
            str(len(self.imgs)) + ": " + self.imgs[self.pos]
        self.lblDir.config(text=label)
        self.posImage = self.pos * 2

    
    def mouse_press(self, event):
        if event.inaxes != self.a:
            return
        self.press = True
        closest_index = self.modificacion.closest(event.xdata, event.ydata)[0]
        self.posicion = [closest_index]
        self.ejeX[closest_index], self.ejeY[closest_index] = event.xdata, event.ydata
        self.cambios()

    def mouse_release(self, event):
        self.press = False

    def mouse_move(self, event):
     if self.press and event.inaxes == self.a:
        # Check if the zoom tool is active
        if self.canvas.toolbar.mode != '':
            # If the zoom tool is active, skip updating the point coordinates
            return

        # Update the coordinates of the selected point
        self.ejeX[self.posicion[0]], self.ejeY[self.posicion[0]] = event.xdata, event.ydata
        self.cambios()


aplicacion1 = Aplicacion()