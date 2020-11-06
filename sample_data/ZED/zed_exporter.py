########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import numpy as np
import sys
from plyfile import PlyData, PlyElement

def main():
    # Create a Camera object
    zed = sl.Camera()
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()

    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type,
                                    depth_mode = sl.DEPTH_MODE.ULTRA,
                                    coordinate_units = sl.UNIT.MILLIMETER,
                                    camera_disable_self_calib=True)
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    image_sl = sl.Mat()
    disp_sl = sl.Mat()

    cam_info = zed.get_camera_information().calibration_parameters


    fx = cam_info.left_cam.fx
    fy = cam_info.left_cam.fy
    cx = cam_info.left_cam.cx
    cy = cam_info.left_cam.cy
    baseline = 120.27

    print(fx)
    print(fy)
    print(cx)
    print(cy)

    name="standard.ply"

    frame=400

    zed.set_svo_position(frame)

    for i in range(0, 2):
        if i > 0:
            zed.set_svo_position(frame)
            runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use STANDARD sensing mode
            name="fill.ply"

        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_sl, sl.VIEW.RIGHT)
            image_sl.write("right/right.png")
            zed.retrieve_image(image_sl, sl.VIEW.LEFT)
            zed.retrieve_measure(disp_sl, sl.MEASURE.DISPARITY)
            image_sl.write("left/left.png")

            image = image_sl.get_data()
            disp = disp_sl.get_data()

            H = disp.shape[0]
            W = disp.shape[1]
            depth = (baseline * fx) / disp
            pts_color_full = []

            for m in range(0, H):
                for n in range(0, W):
                    color = image[m, n]
                    z = depth[m, n]
                    if z < 0.:
                        x = ((n-cx)*z)/fx
                        y = ((m-cy)*z)/fy
                    else:
                        x = 0
                        y = 0
                        z = 0
                    pts_color_full.append((x, y, z, int(color[2]), int(color[1]), int(color[0])))

            vertex = np.array(pts_color_full, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el], text=False).write(name)

            zed.retrieve_measure(disp_sl, sl.MEASURE.DISPARITY, resolution=sl.Resolution(448, 256))
            disp = disp_sl.get_data()
            np.save("disp/disp_hourglass.npy", disp)

            zed.retrieve_measure(disp_sl, sl.MEASURE.DISPARITY, resolution=sl.Resolution(324, 180))
            disp = disp_sl.get_data()
            np.save("disp/disp_stereonet.npy", disp)


    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
