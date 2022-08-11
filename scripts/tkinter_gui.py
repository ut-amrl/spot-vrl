"""Contains graphical user interface for recieving user ranking based off of cluster image grids"""
__author__= "Daniel Farkash"
__email__= "dmf248@cornell.edu"
__date__= "August 10, 2022"

from tkinter import *
from PIL import Image, ImageTk
from tkinter.font import Font

# runs the tkinter GUI
def run_gui(num):

    # create window
    window = Tk()
    window.geometry(str(num*323)+'x500')
    window.title("Image cost assignment")

    spins = []

    # display each cluster image grid
    for i in range(num):

        # place the text for each image on the window
        txt = "  Terrain " + str(i) + "  "
        lbl = Label(window, text=txt, font=("Arial Bold", 25))
        lbl.grid(column=i, row=0)

        # load and place the cluster image grid into the window
        img=Image.open("/home/dfarkash/Documents/assign_cost_gui/img_samples/group"+str(i)+".png")
        
        photo=ImageTk.PhotoImage(img)

        label = Label(image=photo)
        label.photo=photo
        label.grid(row=1, column=i)

        imgs.append(label)

        # add ranking box to the window
        spin = Spinbox(window, from_=0, to=20, width=5, font=Font(family='Helvetica', size=25, weight='bold'))
        spin.grid(column=i,row=2)
        spins.append(spin)

    # on submission button press, create ranking array
    def clicked():

        for i in range(num):
            ranks.append(spins[i].get())

        window.quit()
        
    # add submission butten to window
    btn = Button(window, text="Submit ranking", command=clicked)
    btn.grid(column=int(num/2), row=4)

    prnt = Label(window, text="________", font=("Arial Bold", 25))
    prnt.grid(column=int(num/2), row=3)


    window.mainloop()


if __name__ == '__main__':

# can re-implement below code, but not needed for current function
    # parser = argparse.ArgumentParser(description='Cost assignment')
    # parser.add_argument('--model_loc', type=str, default='version_14', metavar='N',
    #                     help='model checkpoint log location')
    # parser.add_argument('--data_loc', type=str, default='logs/', metavar='N',
    #                     help='data location')  
    # args = parser.parse_args()                  

    imgs=[]
    ranks=[]
    #TODO: change number based on number of saved cluster grid images
    run_gui(7)
    print(ranks)