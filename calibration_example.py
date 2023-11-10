import numpy as np
from sklearn.isotonic import IsotonicRegression

# Load calibration data.
# Note:
# [p_r] contains the expected confidence levels for the red channel.
# [hat_p_r] contains the empirical confidence levels for the red channel.
# The above is true for [p_g], [hat_p_g], [p_b], and [hat_p_b] w.r.t. the relevant color channels as well.

p_r = np.load("./p_r.npy")
hat_p_r = np.load("./hat_p_r.npy")
p_g = np.load("./p_g.npy")
hat_p_g = np.load("./hat_p_g.npy")
p_b = np.load("./p_b.npy")
hat_p_b = np.load("./hat_p_b.npy")

D_R = (p_r, hat_p_r)
D_G = (p_g, hat_p_g)
D_B = (p_b, hat_p_b)

# Fit auxiliary models A^{R}, A^{G}, and A^{B} on calibration sets D^{R}, D^{G}, and D^{B}.
A_R = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_R[0], D_R[1]
)
A_G = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_G[0], D_G[1]
)
A_B = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(
    D_B[0], D_B[1]
)

# Predict the calibrated confidence level [cal_p_r] given the expected confidence level [p_r] for the red color channel.
p_r = 0.8 # Set this to the output of the uncalibrated FlipNeRF CDF.
cal_p_r = A_R.predict([p_r])
print("p_r: " + str(p_r))
print("cal_p_r: " + str(cal_p_r))
