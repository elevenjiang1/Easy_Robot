import os
import sys
import glob
import shutil
import cv2 as cv
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def see_data():
    """
    Basic see record data
    """
    data_path="/home/elevenjiang/Documents/Project/Easy_Robot/code/robot/data/2023-02-17_15-25"
    indexs_files=glob.glob(os.path.join(data_path,"color_*.png"))
    
    all_pose_array=np.load(os.path.join(data_path,"all_poses.npz"))['all_poses']
    for index in range(len(indexs_files)):
        if index==0:#To compute delta xyz
            continue
        print("***************Now is loading:{}/{} data**************************".format(index,len(indexs_files)))
        color_image=cv.imread(os.path.join(data_path,"color_{}.png".format(index)))
        pose=all_pose_array[index]
        pose_before=all_pose_array[index-1]
        print(pose)
        print(pose_before)
        cv.imshow("color_image",color_image)
        cv.waitKey(0)
        
def clean_dataset():
    """
    To delete target_objects
    """
    root_data_path="/home/elevenjiang/Documents/Project/Easy_Robot/code/robot/data"
    all_time_folders=os.listdir(root_data_path)
    all_time_folders.sort()
    
    #1: Load all time folder
    target_root_data_path=os.path.join(Abs_Path,"clean_data")
    for idx,time_folder in enumerate(all_time_folders):
        print("*********************Processing {}/{} folder...*********************".format(idx,len(all_time_folders)))
        data_path=os.path.join(root_data_path,time_folder)
        indexs_files=glob.glob(os.path.join(data_path,"color_*.png"))
        
        #2: Find the xyz_move begin index
        all_pose_array=np.load(os.path.join(data_path,"all_poses.npz"))['all_poses']
        move_index=0
        for index in range(len(indexs_files)):
            if index==0:#To compute delta xyz
                continue
            
            pose=all_pose_array[index]
            pose_before=all_pose_array[index-1]
            xyz_move=np.linalg.norm(pose[:3]-pose_before[:3])
            print("index:{} move is:{:.5}".format(index,xyz_move))
            
            if xyz_move>0.00005:#Thresold for move_flag
                print("Robot begin move in {} in folder {}".format(index,time_folder))
                temp_input=input("To generate new dataset? y for generate n for no")
                if temp_input=='y':
                    move_index=index
                    #3: Begin to move data
                    target_folder=os.path.join(target_root_data_path,time_folder)
                    if os.path.exists(target_folder):
                        print("{} folder exist!!! please check!".format(target_folder))
                        return
                    else:
                        os.mkdir(target_folder)
                    
                    begin_index=max(move_index-5,0)#avoid lower than 0
                    all_count=0
                    pose_list=[]
                    while (all_count+begin_index)<len(indexs_files):
                        pose_list.append(all_pose_array[int(begin_index+all_count)])
                        
                        origin_image_path=os.path.join(data_path,"color_{}.png".format(all_count+begin_index))
                        target_image_path=os.path.join(target_folder,"color_{}.png".format(all_count))
                        shutil.copy(origin_image_path,target_image_path)
                        
                        all_count=all_count+1
                    save_array=np.array(pose_list)
                    np.savez(os.path.join(target_folder, "all_poses.npz"), all_poses=save_array)
                break
                        
                    

class VisionDataset(Dataset):
    def __init__(self,dataset_path,split=None,split_rate=None):
        """
        """
        self.dataset_path=dataset_path
        time_folders_list=os.listdir(self.dataset_path)
        time_folders_list.sort()
        
        if split=='train':
            select_time_folders=(time_folders_list[:int(len(time_folders_list)*split_rate)])
        else:
            select_time_folders=(time_folders_list[int(len(time_folders_list)*split_rate):])
        
        
        self.index_list=[]
        self.pose_dict={}
        for folder in select_time_folders:
            color_image_list=glob.glob(os.path.join(dataset_path,folder,"color_*.png"))
            self.pose_dict[folder]=np.load(os.path.join(dataset_path,folder,"all_poses.npz"))['all_poses']
            self.index_list=self.index_list+color_image_list
        
        
            
        print("Load {} dataset, contain {} data".format(split,len(self.index_list)))
        

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        #1: Load index file
        color_image_path=self.index_list[index]
        image=cv.imread(color_image_path)
        image=cv.resize(image,dsize=(128,128))
        image=np.transpose(image,[2,0,1])
        
        
        folder_path=color_image_path.split('/')[-2]
        color_name=color_image_path.split('/')[-1]
        color_index=int(color_name.split('_')[-1][:-4])
        
        pose=self.pose_dict[folder_path][color_index]
        before_pose=self.pose_dict[folder_path][color_index-1]
        delta_pose=pose-before_pose
        
        return image,delta_pose.astype(np.float32)
        

def example_load_data():
    """
    """
    dataset_path="/home/elevenjiang/Documents/Project/Easy_Robot/code/network/clean_data"
    tactile_dataset=VisionDataset(dataset_path,split='test',split_rate=0.8)
    train_loader=DataLoader(tactile_dataset,batch_size=1,shuffle=False)    

    for data in tactile_dataset:
        image,delta_pose=data

        print(image.shape)
        print(delta_pose.shape)

        break

    for data in train_loader:
        image,delta_pose=data

        print(image.shape)
        print(delta_pose.shape)
        
        print(delta_pose)
        break


if __name__ == '__main__':
    # see_data()
    # clean_dataset()
    example_load_data()




        
