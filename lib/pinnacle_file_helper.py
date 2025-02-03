import datetime

import cv2
import numpy as np


def read_pinnacle_img(pinnacle_folder):
    img_header = {}

    with open(pinnacle_folder + '/img.header') as f:
        for line in f:
            first_split = line.split('=')
            second_split = line.split(':')

            if len(first_split) > len(second_split):
                img_header[first_split[0].strip()] = first_split[1].strip()
            else:
                img_header[second_split[0].strip()] = second_split[1].strip()

    for key, value in img_header.items():
        img_header[key] = value.replace(';', '')

    for key, value in img_header.items():
        img_header[key] = value.replace('"', '')

    for key, value in img_header.items():
        if value.isnumeric():
            img_header[key] = int(value)

    img_header['vol_max'] = float(img_header['vol_max'])
    img_header['vol_min'] = float(img_header['vol_min'])

    img_header['t_pixdim'] = float(img_header['t_pixdim'])
    img_header['x_pixdim'] = float(img_header['x_pixdim'])
    img_header['y_pixdim'] = float(img_header['y_pixdim'])
    img_header['z_pixdim'] = float(img_header['z_pixdim'])

    img_header['t_start'] = float(img_header['t_start'])
    # img_header['x_start'] = float(img_header['x_start'])
    # img_header['y_start'] = float(img_header['y_start'])
    if 'x_start_dicom' in img_header.keys():
        img_header['x_start'] = float(img_header['x_start_dicom'])
    else:
        img_header['x_start'] = float(img_header['x_start'])
    if 'y_start_dicom' in img_header.keys():
        img_header['y_start'] = float(img_header['y_start_dicom'])
    else:
        img_header['y_start'] = float(img_header['y_start'])
    img_header['z_start'] = float(img_header['z_start'])
    img_header['z_time'] = float(img_header['z_time'])

    img_header['couch_pos'] = float(img_header['couch_pos'])
    img_header['couch_height'] = float(img_header['couch_height'])
    img_header['X_offset'] = float(img_header['X_offset'])
    img_header['Y_offset'] = float(img_header['Y_offset'])

    img_img = np.fromfile(pinnacle_folder + 'img.img', dtype=np.dtype('<u2'))
    img_img = img_img.reshape((img_header['z_dim'], img_header['x_dim'], img_header['y_dim']))
    img_img = np.moveaxis(img_img, 0, 2)

    return img_header, img_img


def write_pinnacle_roi(pinnacle_folder, img_header, masks, roi_list, roi_2_roi_id):
    color_list = ['red', 'blue', 'skyblue', 'purple',
                  'forest', 'orange', 'lightblue', 'yellowgreen',
                  'khaki', 'aquamarine', 'teal', 'steelblue',
                  'brown', 'olive', 'lavender', 'maroon'
                                                'seashell', 'SUV3', 'CEqual', 'rainbow1',
                  'rainbow2', 'green', 'yellow', 'lavender',
                  'yellowgreen', 'lightorange', 'grey', 'tomato']

    with open(pinnacle_folder + '/deep_learning.roi', 'w') as f:
        f.write('// Region of Interest file\n')
        f.write('// Data set: ' + img_header['db_name'] + '\n')
        f.write('// File created: ' + datetime.datetime.now().strftime("%c") + '\n')
        f.write('\n')
        f.write('//\n')
        f.write('// Pinnacle Treatment Planning System Version 8.0m \n')
        f.write('// 8.0m \n')
        f.write('//\n')
        f.write('\n')
        f.write(' file_stamp={\n')
        f.write('       write_date: ' + datetime.datetime.now().strftime("%c") + '\n')
        f.write('    write_program: Jiazhou DeepLearning Program\n')
        f.write('    write_version: hn 1.0\n')
        f.write('  }; // End of file_stamp\n')
        for roi_i, roi_name in enumerate(roi_list):

            f.write('//-----------------------------------------------------\n')
            f.write('//  Beginning of ROI: ' + roi_name + '\n')
            f.write('//-----------------------------------------------------\n')
            f.write('\n')
            f.write(' roi={\n')
            f.write('           name: ' + roi_name + '\n')
            f.write('    volume_name: ' + img_header['db_name'] + '\n')
            f.write('stats_volume_name: ' + img_header['db_name'] + '\n')
            f.write('           author:   \n')
            f.write('           organ_name:   \n')
            f.write('           flags =          131072;\n')
            f.write('           color:           ' + color_list[roi_2_roi_id[roi_name]] + '\n')
            f.write('           box_size =       5;\n')
            f.write('           line_2d_width =  2;\n')
            f.write('           line_3d_width =  1;\n')
            f.write('           paint_brush_radius =  0.2;\n')
            f.write('           paint_allow_curve_closing = 1;\n')
            f.write('           curve_min_area =  0.1;\n')
            f.write('           curve_overlap_min =  88;\n')
            f.write('           lower =          800;\n')
            f.write('           upper =          4096;\n')
            f.write('           radius =         0;\n')
            f.write('           density =        1;\n')
            f.write('           density_units:   g/cm^3\n')
            f.write('           override_data =  0;\n')
            f.write('           override_material =  0;\n')
            f.write('           material:        None\n')
            f.write('           invert_density_loading =  0;\n')
            f.write('           volume =         0;\n')
            f.write('           pixel_min =      0;\n')
            f.write('           pixel_max =      0;\n')
            f.write('           pixel_mean =     0;\n')
            f.write('           pixel_std =      0;\n')
            f.write('           bBEVDRROutline = 0;\n')
            f.write('           is_linked =      0;\n')
            f.write('           auto_update_contours =  0;\n')

            mask = np.array(masks[:, :, :, roi_2_roi_id[roi_name]], dtype=int)

            num_curve = 0
            for z in range(mask.shape[0]):
                maskcv = np.array(mask[z, :, :], dtype=np.uint8)
                _, contours, _ = cv2.findContours(maskcv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_curve = num_curve + len(contours)
            f.write('           num_curve = ' + str(num_curve) + ';\n')

            curve_i = 0
            for z in range(mask.shape[0]):
                maskcv = np.array(mask[z, :, :], dtype=np.uint8)
                _, contours, _ = cv2.findContours(maskcv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour_i, contour in enumerate(contours):
                    curve_i = curve_i + 1

                    f.write('//----------------------------------------------------\n')
                    f.write('//  ROI: ' + roi_name + '\n')
                    f.write('//  Curve ' + str(curve_i) + ' of ' + str(num_curve) + '\n')
                    f.write('//----------------------------------------------------\n')
                    f.write('               curve={\n')
                    f.write('                       flags =       131092;\n')
                    f.write('                       block_size =  32;\n')
                    f.write('                       num_points =  ' + str(len(contour) + 1) + ';\n')
                    f.write('points={\n')

                    for i in range(0, len(contour)):
                        vox_x = contour[i][0][0]
                        vox_y = contour[i][0][1]
                        vox_z = z

                        f.write('{:.4f}'.format(img_header['x_start'] +
                                                img_header['x_pixdim'] *
                                                float(vox_x)) + ' ' +
                                '{:.4f}'.format(img_header['y_start'] +
                                                img_header['y_pixdim'] *
                                                (img_header['y_dim'] -
                                                 float(vox_y) + 1)) + ' ' +
                                '{:.2f}'.format(img_header['z_start'] +
                                                img_header['z_pixdim'] *
                                                float(vox_z)) + '\n')

                    f.write('{:.4f}'.format(img_header['x_start'] +
                                            img_header['x_pixdim'] *
                                            float(contour[0][0][0])) + ' ' +
                            '{:.4f}'.format(img_header['y_start'] +
                                            img_header['y_pixdim'] *
                                            (img_header['y_dim'] -
                                             float(contour[0][0][1]) + 1)) + ' ' +
                            '{:.2f}'.format(img_header['z_start'] +
                                            img_header['z_pixdim'] *
                                            float(z)) + '\n')

                    f.write(' };  // End of points for curve ' + str(curve_i) + '\n')
                    f.write('}; // End of curve '+ str(curve_i) + '\n')

            f.write('               surface_mesh={\n')
            f.write('                       allow_adaptation = 1;\n')
            f.write('                       replace_mean_mesh = 0;\n')
            f.write('                       auto_clean_contours = 1;\n')
            f.write('                       auto_smooth_mesh = 1;\n')
            f.write('                       internal_weight = -3.000000;\n')
            f.write('                       surface_weight = 1.000000;\n')
            f.write('                       distance_weight = 2.000000;\n')
            f.write('                       profile_length = 10;\n')
            f.write('                       limit_feature_search_string: Full\n')
            f.write('                       pixel_size = 1.000000;\n')
            f.write('                       feature_weight = 0.010000;\n')
            f.write('                       lower_CT_bound = 0.000000;\n')
            f.write('                       upper_CT_bound = 4095.000000;\n')
            f.write('                       gradient_clipping: Soft\n')
            f.write('                       gradient_direction: Unsigned\n')
            f.write('                       maximum_gradient = 100.000000;\n')
            f.write('                       avoid_CT_range: None\n')
            f.write('                       avoid_CT_threshold = 4095.000000;\n')
            f.write('                       number_of_modes = 0;\n')
            f.write('                       number_of_vertices = 0;\n')
            f.write('                       number_of_triangles = 0;\n')
            f.write('                       max_iterations = 10;\n')
            f.write('                       smooth_between_iterations = 0;\n')
            f.write('                       repair_between_iterations = 1;\n')
            f.write('                       translation_increment_cm = 0.500000;\n')
            f.write('                       rotation_increment = 20.000000;\n')
            f.write('                       magnify_increment = 0.200000;\n')
            f.write('                       sphere_tool_radius = 1.500000;\n')
            f.write('                       gaussian_tool_radius = 3.000000;\n')
            f.write('                       vertex_drag_curvature = 0.700000;\n')
            f.write('                       vertex_drag_neighbor_layers = 3;\n')
            f.write('                       vertices={\n')
            f.write(' };  // End of vertices for surface mesh\n')
            f.write('                       triangles={\n')
            f.write(' };  // End of triangles for surface mesh\n')
            f.write('}; // End of surface_mesh \n')
            f.write('               mean_mesh={\n')
            f.write('                       samples = 1;\n')
            f.write('                       number_of_vertices = 0;\n')
            f.write('                       number_of_triangles = 0;\n')
            f.write('                       vertices={\n')
            f.write(' };  // End of vertices for mean mesh\n')
            f.write('                       triangles={\n')
            f.write(' };  // End of triangles for mean mesh\n')
            f.write('}; // End of mean_mesh \n')
            f.write('        }; // End of ROI ' + roi_name + '\n')


def write_pinnacle_script(pinnacle_folder, DVH):
    # 0:CTV, 1:LF, 2:RF, 3:BLADDER, 4:PTV
    LF_DVH = []
    RF_DVH = []
    BLADDER_DVH = []
    for index in range(len(DVH)):
        if index == 1:
            LF_DVH.append(int(DVH[index][150] / DVH[index][0] * 100))
            LF_DVH.append(int(DVH[index][200] / DVH[index][0] * 100))
            LF_DVH.append(int(DVH[index][250] / DVH[index][0] * 100))
        if index == 2:
            RF_DVH.append(int(DVH[index][150] / DVH[index][0] * 100))
            RF_DVH.append(int(DVH[index][200] / DVH[index][0] * 100))
            RF_DVH.append(int(DVH[index][250] / DVH[index][0] * 100))
        if index == 3:
            BLADDER_DVH.append(int(DVH[index][250] / DVH[index][0] * 100))
            BLADDER_DVH.append(int(DVH[index][300] / DVH[index][0] * 100))
            BLADDER_DVH.append(int(DVH[index][350] / DVH[index][0] * 100))
            BLADDER_DVH.append(int(DVH[index][400] / DVH[index][0] * 100))
            BLADDER_DVH.append(int(DVH[index][450] / DVH[index][0] * 100))

    # write script file
    with open(pinnacle_folder + '/auto_plan.Script', 'w') as f:
        f.write('WindowList .CTSim .PanelList .#"#3" .GotoPanel = "FunctionLayoutIcon3";\n')
        f.write('TrialList .Current .LaserLocalizer .LockJaw = "0";\n')
        f.write('ViewWindowList .#"*" .CineOnOff = "0";\n')
        f.write('WindowList .CTSim .PanelList .#"#4" .GotoPanel = "FunctionLayoutIcon4";\n')
        f.write('ViewWindowList .#"*" .CineOnOff = "0";\n')
        f.write('WindowList .TrialPrescription .Create = "Edit Prescriptions...";\n')
        f.write('WindowList .PrescriptionEditor .Create = "Edit...";\n')
        f.write('TrialList .Current .PrescriptionList .Current .PrescriptionDose = "200";\n')
        f.write('TrialList .Current .PrescriptionList .Current .PrescriptionPercent = "96";\n')
        f.write('TrialList .Current .PrescriptionList .Current .NumberOfFractions = "25";\n')
        f.write('TrialList .Current .PrescriptionList .Current .NormalizationMethod = "ROI Mean";\n')
        f.write('TrialList .Current .PrescriptionList .Current .PrescriptionRoi = "PTV";\n')
        f.write('WindowList .PrescriptionEditor .Unrealize = "Dismiss";\n')
        f.write('WindowList .TrialPrescription .Unrealize = "Dismiss";\n')
        f.write('WindowList .BeamWeighting .Create = "Beam Weighting...";\n')
        f.write('WindowList .WeightingOptions .Create = "Weighting Options...";\n')
        f.write('TrialList .Current .WeightEqual = "Set Equal Weights for Unlocked Beams";\n')
        f.write('WindowList .WeightingOptions .Unrealize = "Dismiss";\n')
        f.write('WindowList .BeamWeighting .Unrealize = "Dismiss";\n')
        f.write('WindowList .CTSim .PanelList .#"#4" .GotoPanel = "FunctionLayoutIcon4";\n')
        f.write('ViewWindowList .#"*" .CineOnOff = "0";\n')
        f.write('IF .ControlPanel .Icon .#"Dose Grid Rescale" .Current .THEN .TrialList .Current .DoseGrid .Display2d '
                '= TrialList .Current .DoseGrid .#"!Display2d";\n')
        f.write('IF .ControlPanel .Icon .#"Dose Grid Rescale" .#"!Current" .THEN .TrialList .Current .DoseGrid .'
                'Display2d = "1";\n')
        f.write('ControlPanel .Icon .#"Dose Grid Rescale" .MakeCurrent = "DoseGridPlaceIcon";\n')
        f.write('WindowList .CTSim .PanelList .#"#5" .GotoPanel = "FunctionLayoutIcon5";\n')
        f.write('ViewWindowList .#"*" .CineOnOff = "0";\n')
        f.write('WindowList .PlanEval .CreateUnrealized = "Dose Volume Histogram...";\n')
        f.write('WindowList .PlanEval .PanelList .#"#0" .GotoPanel = "Dose Volume Histogram...";\n')
        f.write('WindowList .PlanEval .Create = "Dose Volume Histogram...";\n')
        f.write('PluginManager .PlanEvalPlugin .TrialList .#"#0" .Selected = 1;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#1" .Selected = 0;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#1" .Selected = 1;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#2" .Selected = 0;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#2" .Selected = 1;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#3" .Selected = 0;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#3" .Selected = 1;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#4" .Selected = 0;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#4" .Selected = 1;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#5" .Selected = 0;\n')
        f.write('PluginManager .PlanEvalPlugin .ROIList .#"#5" .Selected = 1;\n')
        f.write('WindowList .TechniqueEditor .Create = "Technique Window...";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .IsoCenter .ExpectedPoi .Name = "ISO";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .IsoCenter .IsCreatingNewPOI = PluginManager .TTPlugin .'
                'TechMgr .Technique .IsoCenter .ExpectedPoi .#"!IsMapped";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .PrescriptionSpecs .PrescriptionDose = "200";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .PrescriptionSpecs .PrescriptionPercentage = "96";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .PrescriptionSpecs .NormalizationMethod = "ROI Mean";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .PrescriptionSpecs .ExpectedRoi .Name = "PTV";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .PrescriptionSpecs .NumberOfFractions = "25";\n')
        f.write('WindowList .TechniqueEditor .PanelList .#"#1" .GotoPanel = "FunctionLayoutIcon2";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .ConfirmAllBeamDelete = "DMPO";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .MachineName = "VAR5685";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddBeamSpecs = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#0" .GantryStart = "180";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#1" .GantryStart = "230";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#2" .GantryStart = "280";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#3" .GantryStart = "320";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#4" .GantryStart = "40";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#5" .GantryStart = "80";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .BeamSpecsList .#"#6" .GantryStart = "130";\n')
        f.write('TechniqueEditorLayout .Index = 1;\n')
        f.write('WindowList .TechniqueEditor .PanelList .#"#2" .GotoPanel = "FunctionLayoutIcon1";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddTargetOptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .TargetOptGoalList .#"#0" .ExpectedRoi .Name = "PTV";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .TargetOptGoalList .#"#0" .Dose = " 5000";\n')
        # OAR object function
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .AddOAROptGoal = "Add";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#0" .ExpectedRoi .Name = "BLADDER";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#1" .ExpectedRoi .Name = "BLADDER";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#2" .ExpectedRoi .Name = "BLADDER";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#3" .ExpectedRoi .Name = "BLADDER";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#4" .ExpectedRoi .Name = "BLADDER";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#5" .ExpectedRoi .Name = "RF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#6" .ExpectedRoi .Name = "RF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#7" .ExpectedRoi .Name = "RF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#8" .ExpectedRoi .Name = "LF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#9" .ExpectedRoi .Name = "LF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#10" .ExpectedRoi .Name = "LF";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#0" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#1" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#2" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#3" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#4" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#5" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#6" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#7" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#8" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#9" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#10" .ObjectiveType = "Max DVH";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#0" .Dose = "2500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#0" .MaxDVHVolume = "'
                + str(BLADDER_DVH[0]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#1" .Dose = "3000";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#1" .MaxDVHVolume = "'
                + str(BLADDER_DVH[1]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#2" .Dose = "3500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#2" .MaxDVHVolume = "'
                + str(BLADDER_DVH[2]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#3" .Dose = "4000";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#3" .MaxDVHVolume = "'
                + str(BLADDER_DVH[3]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#4" .Dose = "4500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#4" .MaxDVHVolume = "'
                + str(BLADDER_DVH[4]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#5" .Dose = "1500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#5" .MaxDVHVolume = "'
                + str(RF_DVH[0]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#6" .Dose = "2000";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#6" .MaxDVHVolume = "'
                + str(RF_DVH[1]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#7" .Dose = "2500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#7" .MaxDVHVolume = "'
                + str(RF_DVH[2]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#8" .Dose = " 1500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#8" .MaxDVHVolume = "'
                + str(LF_DVH[0]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#9" .Dose = " 2000";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#9" .MaxDVHVolume = "'
                + str(LF_DVH[1]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#10" .Dose = " 2500";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#10" .MaxDVHVolume = "'
                + str(LF_DVH[2]) + '";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#0" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#1" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#2" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#3" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#4" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#5" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#6" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#7" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#8" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#9" .Priority = "Low";\n')
        f.write('PluginManager .TTPlugin .TechMgr .Technique .OAROptGoalList .#"#10" .Priority = "Low";\n')

        f.write('\n')
        f.write('/* ? */\n')
