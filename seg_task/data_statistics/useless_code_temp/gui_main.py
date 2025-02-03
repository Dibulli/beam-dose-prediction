import logging
import os
import sys
from tkinter import Button
from tkinter import Entry
from tkinter import Label
from tkinter import StringVar
from tkinter import Tk
from tkinter import filedialog

from hydra.experimental import compose, initialize

from auto_seg import run_auto_seg

# set some globe running variables
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../..')))
os.environ['ROOT_DIR'] = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

initialize(
    config_dir="conf_sfhz", strict=True,
)


def img_path_selection():
    path_ = filedialog.askdirectory(initialdir="C:/", title="Select DICOM image directory")
    img_path.set(path_)


def des_path_selection():
    path_ = filedialog.askdirectory(initialdir="C:/", title="Select output directory")
    des_path.set(path_)


def head_neck():
    cfg = compose("seg_hn_local.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def head_neck_parotid():
    cfg = compose("seg_hn_local_50sample_parotid.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def head_neck_eye():
    cfg = compose("seg_hn_local_70sample_eyes.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def head_neck_ear():
    cfg = compose("seg_hn_local_60sample_ear.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def head_neck_layrnx():
    cfg = compose("seg_hn_local_layrnx.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def male():
    cfg = compose("seg_male_local.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def chest():
    cfg = compose("seg_chest_local.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


def female():
    cfg = compose("seg_female_local.yaml")
    run_auto_seg(cfg, dcm_img_folder=img_path.get(), output_folder=des_path.get())


root = Tk()
root.geometry('800x500')
root.title("Auto segmentation tool")

img_path = StringVar()
des_path = StringVar()

img_path.set('C:/')
des_path.set('C:/')

Label(root, text="DICOM path:").grid(row=0, column=0)
Entry(root, textvariable=img_path, width=92).grid(row=0, column=1)
Button(root, text="selection", command=img_path_selection).grid(row=0, column=2)

Label(root, text="Output path:").grid(row=1, column=0)
Entry(root, textvariable=des_path, width=92).grid(row=1, column=1)
Button(root, text="selection", command=des_path_selection).grid(row=1, column=2)

Button(root, text="head_neck", command=head_neck, width=20).grid(row=2, column=1)
Button(root, text="chest", command=chest, width=20).grid(row=3, column=1)
Button(root, text="male", command=male, width=20).grid(row=4, column=1)
Button(root, text="female", command=female, width=20).grid(row=5, column=1)
Button(root, text="head_neck_parotid", command=head_neck_parotid, width=20).grid(row=6, column=1)
Button(root, text="head_neck_eye", command=head_neck_eye, width=20).grid(row=7, column=1)
Button(root, text="head_neck_ear", command=head_neck_ear, width=20).grid(row=8, column=1)
Button(root, text="head_neck_layrnx", command=head_neck_layrnx, width=20).grid(row=8, column=1)

root.mainloop()
