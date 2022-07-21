from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cluster_jackal
import random


def run_gui(num):
    window = Tk()

    window.geometry('700x500')

    window.title("Image cost assignment")

    spins = []

    for i in range(num):
        txt = "  Terrain " + str(i) + "  "
        lbl = Label(window, text=txt, font=("Arial Bold", 25))

        lbl.grid(column=i, row=0)

        img=Image.open("/home/dfarkash/Documents/garbage/img"+str(i+2)+".png")
        photo=ImageTk.PhotoImage(img)

        label = Label(image=photo)
        label.photo=photo
        label.grid(row=1, column=i)
        
        # cv = Canvas(window, width=100, height=100)
        # cv.create_image(5, 5, image=photo, anchor='nw')
        # cv.grid(column= i, row = 1)

        imgs.append(label)

        spin = Spinbox(window, from_=0, to=20, width=5)

        spin.grid(column=i,row=2)
        spins.append(spin)


    def clicked():

        res = "Ranking: " + spins[0].get()

        prnt.configure(text= res)


        for i in range(num):
            ranks.append(spins[i].get())

        window.quit()
        

    btn = Button(window, text="Submit ranking", command=clicked)

    btn.grid(column=0, row=3)

    prnt = Label(window, text="", font=("Arial Bold", 25))

    prnt.grid(column=0, row=4)


    window.mainloop()

def sample_clusters(data, visual_patch):
        clusters , elbow = cluster_jackal.cluster(data)
        dic = {}
        for i in range(elbow):
            dic[i] = []
        for i in range(elbow):
            idx = np.where(clusters ==i)
            for j in range(25):
                chosen = np.random.randint(0, len(idx))
                visual_patch = visual_patch[idx[chosen], :, :, :]
                dic[i].append(visual_patch)

        return dic, elbow

def img_clusters(dic, elbow):
    for i in range(elbow):
        new_im = Image.new('RGB', (3000,3000))
        for j in range(25):
            visual_patch = dic[i][j]
            visual_patch = visual_patch.cpu()
            visual_patch = visual_patch.numpy()
            h = int(j/5)
            w = j%5
            im = im.fromarray(visual_patch)
            im.thumbnail((300,300))
            new_im.paste(im, (h,w))
        new_im.save("/home/dfarkash/garbage" +"/group"+str(j)+".png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cost assignment')
    parser.add_argument('--model_loc', type=str, default='version_14', metavar='N',
                        help='model checkpoint log location')
    parser.add_argument('--data_loc', type=str, default='logs/', metavar='N',
                        help='data location')  
    args = parser.parse_args()                  

    model = MyLightningModule.load_from_checkpoint(args.model_loc)
    model.eval()
    dataset = CustomDataset(args.data_loc)
    visual_patches = []
    for i in range(2000):
        rand = random.randint(0,dataset.__len__()-1)
        main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = dataset.__getitem__(rand)
        visual_patch.append(main_patch[0])

    

    img_clusters(sample_clusters(data, visual_patches))

    imgs=[]
    ranks=[]
    run_gui(7)
    print(ranks)