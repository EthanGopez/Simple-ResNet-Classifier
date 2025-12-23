import argparse
import csv

import numpy as np
import cv2

def generate(
        dir : str,
        image_count : int = 2000,
        lesion_prop : float = 0.1,
        height : int = 28,
        width : int = 28,
        seed : int = 42
):
    rng = np.random.default_rng(seed=seed)
    dir = dir + "/"

    # hard-coded parameters
    START_ANGLE = 0
    END_ANGLE = 360
    COLOR_CHANNELS = 3
    WHITE = (255, 255, 255) # White in BGR format (Blue, Green, Red)
    BLACK = (0, 0, 0)
    THICKNESS = -1

    # liver hyperparameters
    c_low_bound = 0.4
    c_up_bound = 0.6
    a_low = 0.4
    a_high = 0.5
    b_low = 0.2
    b_high = 0.3
    theta_low = 0.0
    theta_high = 360

    # lesion hyperparameters
    scale_factor_low =  0.1
    scale_factor_high = 0.4
    
    lesion_prop_boundary: float = image_count * lesion_prop

    with open(dir + 'labels.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(image_count):
            file_name = str(i).zfill(4) + ".png"

            img = np.zeros((height, width, COLOR_CHANNELS), dtype=np.uint8)
            x_c = rng.uniform(low=c_low_bound, high=c_up_bound) * width
            y_c = rng.uniform(low=c_low_bound, high=c_up_bound) * height
            a = rng.uniform(low=a_low, high=a_high) * width
            b = rng.uniform(low=b_low, high=b_high) * height
            theta = rng.uniform(low=theta_low, high=theta_high)

            center_coordinates = (round(x_c), round(y_c))
            axes_length = (round(a), round(b)) # Half the width and half the height of the ellipse

            # Draw the filled ellipse
            cv2.ellipse(img, center_coordinates, axes_length, theta, START_ANGLE, END_ANGLE, WHITE, THICKNESS)

            if i < lesion_prop_boundary:
                # case: we have a lesion
                writer.writerow([file_name, 1])

                # logic assisted with Google Gemini; however code written is
                # entirely my own
                
                # parameters will need to be scaled to existing ellipse dimensions
                scale_factor = rng.uniform(low=scale_factor_low, high=scale_factor_high)
                inner_a = a * scale_factor
                inner_b = b * scale_factor
                inner_axes_length = (round(inner_a), round(inner_b))

                # define a half-sized "valid area" for the center of the lesion to be placed
                center_diff_a = (a - inner_a) / 2.0
                center_diff_b = (b - inner_b) / 2.0
                # use rejection sampling to figure out a valid lesion location
                found_unit_circle = False
                while not found_unit_circle:
                    x_scale = rng.uniform(low=-1, high=1)
                    y_scale = rng.uniform(low=-1, high=1)
                    found_unit_circle = x_scale**2 + y_scale**2 <= 1
                
                inner_center_coordinates = (
                    round(x_c + x_scale * center_diff_a),
                    round(y_c + y_scale * center_diff_b)
                )

                inner_theta = rng.uniform(low=theta_low, high=theta_high)
                cv2.ellipse(img, inner_center_coordinates, inner_axes_length, inner_theta, START_ANGLE, END_ANGLE, BLACK, THICKNESS)
                
            else:
                # case: we do NOT have a lesion
                writer.writerow([file_name, 0])

            cv2.imwrite(dir + file_name, img)


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--dir", type=str, required=True) #if not train set, then val set
    
    parser.add_argument("--image_count", type=int, default=2000)
    parser.add_argument("--lesion_prop", type=float, default=0.2)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--width", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)

    generate(**vars(parser.parse_args()))