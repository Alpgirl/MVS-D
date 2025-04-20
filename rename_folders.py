import os
from datetime import datetime

root_dir = "/beegfs/home/i.larina/MVS-D/MVSFormerPlusPlus/saved/models/DINOv2/MVSD++_train_20250408_032611/test_nv10_prob0.6/_2368x1920/"

target_name = "consistencyCheck-20250417"

for subdir in os.listdir(root_dir):
    root_subdir = os.path.join(root_dir, subdir, "points_mvsnet")
    folders = os.listdir(root_subdir)

    if len(folders) > 1:
        times = []
        for folder in folders:
            date_str, time_str = folder.split('-')[1], folder.split('-')[2]
            str = date_str + " " + time_str
            folder_time = datetime.strptime(str, "%Y%m%d %H%M%S")
            times.append(folder_time)
        ord = times.index(max(times))
        folders = [folders[ord]]
    
    item_path = os.path.join(root_subdir, folders[0])
    if os.path.isdir(item_path) and item_path.split('/')[-1].startswith("consistencyCheck"):
        new_path = os.path.join(root_subdir, target_name)
        print(item_path, new_path)

        try:
            os.rename(item_path, new_path)
        except OSError as e:
            print('Error renaming {0}: {1}'.format(item_path, e))
    
print("Renaming complete.")