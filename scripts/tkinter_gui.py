from tkinter import *
from PIL import Image, ImageTk
from tkinter.font import Font


def run_gui(num):
    window = Tk()

    window.geometry(str(num*323)+'x500')

    window.title("Image cost assignment")

    spins = []

    for i in range(num):
        txt = "  Terrain " + str(i) + "  "
        lbl = Label(window, text=txt, font=("Arial Bold", 25))

        lbl.grid(column=i, row=0)

        img=Image.open("/home/dfarkash/Documents/assign_cost_gui/img_samples/group"+str(i)+".png")
        photo=ImageTk.PhotoImage(img)

        label = Label(image=photo)
        label.photo=photo
        label.grid(row=1, column=i)
        
        # cv = Canvas(window, width=100, height=100)
        # cv.create_image(5, 5, image=photo, anchor='nw')
        # cv.grid(column= i, row = 1)

        imgs.append(label)

        spin = Spinbox(window, from_=0, to=20, width=5, font=Font(family='Helvetica', size=25, weight='bold'))

        spin.grid(column=i,row=2)
        spins.append(spin)


    def clicked():

        # res = "Ranking: " + spins[0].get()

        # prnt.configure(text= res)


        for i in range(num):
            ranks.append(spins[i].get())

        window.quit()
        

    btn = Button(window, text="Submit ranking", command=clicked)

    btn.grid(column=int(num/2), row=4)

    prnt = Label(window, text="________", font=("Arial Bold", 25))

    prnt.grid(column=int(num/2), row=3)


    window.mainloop()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Cost assignment')
    # parser.add_argument('--model_loc', type=str, default='version_14', metavar='N',
    #                     help='model checkpoint log location')
    # parser.add_argument('--data_loc', type=str, default='logs/', metavar='N',
    #                     help='data location')  
    # args = parser.parse_args()                  

    imgs=[]
    ranks=[]
    run_gui(7)
    print(ranks)