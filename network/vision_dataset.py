import os
import sys
import glob
import shutil
import random
import cv2 as cv
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

Abs_Path = os.path.dirname(os.path.abspath(__file__))


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qx, qy, qz, qw])


def see_data():
    """
    Basic see record data
    """
    data_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/robot/data/2023-02-17_15-25"
    indexs_files = glob.glob(os.path.join(data_path, "color_*.png"))

    all_pose_array = np.load(os.path.join(
        data_path, "all_poses.npz"))['all_poses']
    for index in range(len(indexs_files)):
        if index == 0:  # To compute delta xyz
            continue
        print("***************Now is loading:{}/{} data**************************".format(index, len(indexs_files)))
        color_image = cv.imread(os.path.join(
            data_path, "color_{}.png".format(index)))
        pose = all_pose_array[index]
        pose_before = all_pose_array[index-1]
        print(pose)
        print(pose_before)
        cv.imshow("color_image", color_image)
        cv.waitKey(0)


def clean_dataset():
    """
    To delete target_objects
    """
    root_data_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/robot/data"
    all_time_folders = os.listdir(root_data_path)
    all_time_folders.sort()

    # 1: Load all time folder
    target_root_data_path = os.path.join(Abs_Path, "clean_data")
    for idx, time_folder in enumerate(all_time_folders):
        print("*********************Processing {}/{} folder...*********************".format(idx, len(all_time_folders)))
        data_path = os.path.join(root_data_path, time_folder)
        indexs_files = glob.glob(os.path.join(data_path, "color_*.png"))

        # 2: Find the xyz_move begin index
        all_pose_array = np.load(os.path.join(
            data_path, "all_poses.npz"))['all_poses']
        move_index = 0
        for index in range(len(indexs_files)):
            if index == 0:  # To compute delta xyz
                continue

            pose = all_pose_array[index]
            pose_before = all_pose_array[index-1]
            xyz_move = np.linalg.norm(pose[:3]-pose_before[:3])
            print("index:{} move is:{:.5}".format(index, xyz_move))

            if xyz_move > 0.00005:  # Thresold for move_flag
                print("Robot begin move in {} in folder {}".format(
                    index, time_folder))
                temp_input = input(
                    "To generate new dataset? y for generate n for no")
                if temp_input == 'y':
                    move_index = index
                    # 3: Begin to move data
                    target_folder = os.path.join(
                        target_root_data_path, time_folder)
                    if os.path.exists(target_folder):
                        print("{} folder exist!!! please check!".format(
                            target_folder))
                        return
                    else:
                        os.mkdir(target_folder)

                    begin_index = max(move_index-5, 0)  # avoid lower than 0
                    all_count = 0
                    pose_list = []
                    while (all_count+begin_index) < len(indexs_files):
                        pose_list.append(
                            all_pose_array[int(begin_index+all_count)])

                        origin_image_path = os.path.join(
                            data_path, "color_{}.png".format(all_count+begin_index))
                        target_image_path = os.path.join(
                            target_folder, "color_{}.png".format(all_count))
                        shutil.copy(origin_image_path, target_image_path)

                        all_count = all_count+1
                    save_array = np.array(pose_list)
                    np.savez(os.path.join(target_folder,
                             "all_poses.npz"), all_poses=save_array)
                break


def sort_images(x):
    color_file_name = x.split('/')[-1]
    color_index = color_file_name.split('_')[-1]
    color_index = int(color_index[:-4])
    return color_index


def move_to_big_dataset():
    target_move_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/network/big_dataset"
    if not os.path.exists(target_move_path):
        os.mkdir(target_move_path)
    else:
        print("Exist {} folder!!!,please check".format(target_move_path))
        return

    clean_data_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/network/clean_data"
    folders_list = os.listdir(clean_data_path)

    save_count = 0
    all_delta_pose_list = []
    for folder in folders_list:
        print("Processing {} folder...".format(folder))
        folder_path = os.path.join(clean_data_path, folder)
        all_pose_array = np.load(os.path.join(
            folder_path, "all_poses.npz"))['all_poses']

        all_color_files = glob.glob(os.path.join(folder_path, "color_*.png"))
        for index in range(len(all_color_files)):
            if index+1 >= len(all_color_files):
                break

            # save color image
            color_image_path = os.path.join(
                folder_path, "color_{}.png".format(index))
            target_color_image_path = os.path.join(
                target_move_path, "color_{}.png".format(save_count))
            shutil.copy(color_image_path, target_color_image_path)

            # save delta pose
            pose_after=all_pose_array[index+1]
            pose=all_pose_array[index]
            quaternion_after=get_quaternion_from_euler(roll=pose_after[3],pitch=pose_after[4],yaw=pose_after[5])
            quaternion=get_quaternion_from_euler(roll=pose[3],pitch=pose[4],yaw=pose[5])
            
            pose_after=np.concatenate([pose_after[:3],quaternion_after])
            pose=np.concatenate([pose[:3],quaternion])
            
            
            delta_pose = pose_after-pose
            all_delta_pose_list.append(delta_pose)
            save_count = save_count+1

    save_delta_array = np.array(all_delta_pose_list)
    np.savez(os.path.join(target_move_path, "all_delta_poses.npz"),all_poses=save_delta_array)


class VisionDataset(Dataset):
    def __init__(self, dataset_path, split=None, split_rate=None):
        """
        """
        # Example for shuffle folders
        # self.dataset_path=dataset_path
        # time_folders_list=os.listdir(self.dataset_path)
        # time_folders_list.sort()
        # if split=='train':
        #     select_time_folders=(time_folders_list[:int(len(time_folders_list)*split_rate)])
        # else:
        #     select_time_folders=(time_folders_list[int(len(time_folders_list)*split_rate):])
        # self.index_list=[]
        # self.pose_dict={}
        # for folder in select_time_folders:
        #     color_image_list=glob.glob(os.path.join(dataset_path,folder,"color_*.png"))
        #     self.pose_dict[folder]=np.load(os.path.join(dataset_path,folder,"all_poses.npz"))['all_poses']
        #     self.index_list=self.index_list+color_image_list

        # Example for load sorted folders save_path
        # self.index_list=[]
        # self.pose_dict={}
        # for folder in time_folders_list:
        #     color_image_list=glob.glob(os.path.join(dataset_path,folder,"color_*.png"))
        #     color_image_list=sorted(color_image_list,key=sort_images)
        #     if split=='train':
        #         for path in color_image_list[:int(len(color_image_list)*split_rate)]:
        #             self.index_list.append(path)
        #     else:
        #         for path in color_image_list[:int(len(color_image_list)*split_rate)]:
        #             self.index_list.append(path)

        # Example for load all big dataset
        all_color_images_index = glob.glob(os.path.join(dataset_path, "color_*.png"))
        random.shuffle(all_color_images_index)
        self.all_pose_array = np.load(os.path.join(
            dataset_path, "all_delta_poses.npz"))['all_poses']

        if split == 'train':
            self.index_list = all_color_images_index[:int(
                len(all_color_images_index)*split_rate)]
        else:
            self.index_list = all_color_images_index[int(
                len(all_color_images_index)*split_rate):]

        print("Load {} dataset, contain {} data".format(
            split, len(self.index_list)))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        # 1: Load index file
        # example for load diff folder
        # color_image_path=self.index_list[index]
        # image=cv.imread(color_image_path)
        # image=cv.resize(image,dsize=(128,128))
        # image=np.transpose(image,[2,0,1])

        # folder_path=color_image_path.split('/')[-2]
        # color_name=color_image_path.split('/')[-1]
        # color_index=int(color_name.split('_')[-1][:-4])

        # pose=self.pose_dict[folder_path][color_index]
        # before_pose=self.pose_dict[folder_path][color_index-1]
        # delta_pose=pose-before_pose

        # 2: Load differ files
        color_image_path = self.index_list[index]
        image = cv.imread(color_image_path)
        image=cv.resize(image,dsize=(128,128))
        image=np.transpose(image,[2,0,1])

        color_file_name = color_image_path.split('/')[-1]
        color_index = color_file_name.split('_')[-1]
        color_index = int(color_index[:-4])
        delta_pose = self.all_pose_array[color_index]
        
        return image, delta_pose.astype(np.float32)


def example_load_data():
    """
    """
    dataset_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/network/clean_data"
    dataset_path = "/home/elevenjiang/Documents/Project/Easy_Robot/code/network/big_dataset"
    
    tactile_dataset = VisionDataset(dataset_path, split='test', split_rate=0.8)
    train_loader = DataLoader(tactile_dataset, batch_size=1, shuffle=False)

    for data in tactile_dataset:
        image, delta_pose = data

        print(image.shape)
        print(delta_pose.shape)

        break

    for data in train_loader:
        image, delta_pose = data

        print(image.shape)
        print(delta_pose.shape)

        print(delta_pose)
        break


if __name__ == '__main__':
    # see_data()
    # clean_dataset()
    # example_load_data()
    move_to_big_dataset()
