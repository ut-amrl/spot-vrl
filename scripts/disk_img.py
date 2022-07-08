
from PIL import Image
import pickle
import argparse
import os


def to_disk(pickle_file_path, save_file_path):
    pickle_file_name = pickle_file_path.split('/')[-1].split('.')[0]
    label = pickle_file_path.split('/')[-2]
    part_save_path = save_file_path+label+"/"
    folder_path = os.path.join(part_save_path,pickle_file_name)
    print(folder_path)
    #os.mkdir(folder_path)
    full_save_path = part_save_path + pickle_file_name+"/"
    data = pickle.load(open(pickle_file_path, 'rb'))
    print("loaded")
    inertial_data = data['imu_jackal']
    pickle.dump(inertial_data, open(full_save_path + "inertial_data.pkl", 'wb'))
    print('dumped"')
    print('length:'+str(len(data['patches'])))
    for i in range(len(data['patches'])):
        if i%1000 = 0:
            print("reached:"+str(i))
        dic = data['patches'][i]
        for j in dic.keys():
            lst = dic[j]
            for k in range(len(lst)):
                im = Image.fromarray(lst[k])
                img_name = str(i) +"_"+ str(j) +"_"+ str(k) +".png"
                img_path = full_save_path + img_name
                im.save(img_path)
    print("finished")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save to disk')
    parser.add_argument('--input', type=str, default='./concrete/sample_data.pkl', metavar='N', help='input location')
    parser.add_argument('--out', type=str,default='./',metavar='N',help='save location')
    args = parser.parse_args()
    to_disk(args.input,args.out)