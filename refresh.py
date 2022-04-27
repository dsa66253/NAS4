import shutil
import os

deleteDirList = ["./nasSavedModel",
                "./tensorboard_pdarts_nodrop",
                "./savedCheckPoint",
                "./saved_mask_per_epoch",
                "./weights_pdarts_nodrop"
                ]

for folder in deleteDirList:
    if os.path.exists(folder):
        shutil.rmtree(folder)