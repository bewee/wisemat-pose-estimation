class constants:
    MATTRESS_FORMAT = None
    SENSORS_X = None
    SENSORS_Y = None
    SKELETON_FORMAT = None
    LABELS = None
    JOINTS = None
    PARTS = None
    MATTRESS_WIDTH = 0.762
    MATTRESS_HEIGHT = 1.85
    MATTRESS_WIDTH_CM = 0.762 * 100
    MATTRESS_HEIGHT_CM = 1.85 * 100

def set_mattress_format(format):
    global constants
    if format == 'clever' or format == 'hrl-ros':
        constants.SENSORS_X = 27
        constants.SENSORS_Y = 64
    elif format == 'softline':
        constants.SENSORS_X = 26
        constants.SENSORS_Y = 64
    elif format == 'slp':
        constants.SENSORS_X = 84
        constants.SENSORS_Y = 192
    else:
        raise "Invalid format"
    constants.MATTRESS_FORMAT = format

def set_skeleton_format(format):
    global constants
    if format == 'clever':
        constants.LABELS = ['Pelvis', 'L Hip', 'R Hip', 'Spine 1', 'L Knee', 'R Knee',
                            'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot', 'R Foot',
                            'Neck', 'L Sh.in', 'R Sh.in', 'Head', 'L Sh.ou', 'R Sh.ou',
                            'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand', 'R Hand']
    elif format == 'hrl-ros':
        constants.LABELS = ['Head', 'Chest', 'R-Elbow', 'L-Elbow', 'R-Wrist',
                            'L-Wrist', 'R-Knee', 'L-Knee', 'R-Ankle', 'L-Ankle']
    elif format == 'slp':
        constants.LABELS = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 
                            'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist', 'Thorax', 'Head top']           
    elif format == 'common':
        constants.LABELS = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 
                            'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow', 'Left wrist', 'Thorax']           
        constants.PARTS = ['Right lower leg', 'Right upper leg', 'Left upper leg', 'Left lower leg',
                           'Right forearm', 'Right upper arm', 'Left upper arm', 'Left forearm']
    elif format == 'arms-omitted':
        constants.LABELS = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee',
                            'Left ankle', 'Right shoulder', 'Left shoulder', 'Thorax']           
        constants.PARTS = ['Right lower leg', 'Right upper leg', 'Left upper leg', 'Left lower leg']
    else:
        raise "Invalid format"
    constants.JOINTS = len(constants.LABELS)
    constants.SKELETON_FORMAT = format

def set_format(format):
    set_mattress_format(format)
    set_skeleton_format(format)
    
set_mattress_format('clever')
set_skeleton_format('common')
